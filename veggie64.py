bl_info = {
    "name": "gl64export",
    "author": "Dragorn421",
    "version": (1, 0),
    "blender": (3, 6, 1),
    "location": "",
    "description": "",
    "category": "Import-Export",
}

print("Hey from", __file__)

# SPDX-License-Identifier: GPL-3.0-or-later

import typing
import dataclasses
from pathlib import Path
import abc
import functools
import math

import bpy


class CWriter:
    class CWriterIndentedContext:
        def __init__(self, cwriter: "CWriter"):
            self.cwriter = cwriter

        def __enter__(self):
            self.cwriter.indent_add()

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.cwriter.indent_remove()

    def __init__(self, write: typing.Callable[[str], None], indent=" " * 4):
        self._write = write
        self._is_at_line_start = True
        self._indent = indent
        self._cur_indent = ""
        self._indent_level = 0

    def _set_indent_level(self, indent_level: int):
        self._indent_level = indent_level
        self._cur_indent = self._indent * indent_level

    def indent_add(self):
        self._set_indent_level(self._indent_level + 1)

    def indent_remove(self):
        self._set_indent_level(self._indent_level - 1)

    def indented(self):
        return CWriter.CWriterIndentedContext(self)

    def write(self, s: str):
        if self._is_at_line_start:
            self._write(self._cur_indent)
            self._is_at_line_start = False
        lines = s.splitlines(keepends=True)
        self._write(self._cur_indent.join(lines))
        if lines and lines[-1].endswith("\n"):
            self._is_at_line_start = True

    def writeln(self, line: str):
        self.write(line + "\n")


def object_set_mode(context, object: bpy.types.Object, mode: str):
    with context.temp_override(object=object):
        bpy.ops.object.mode_set(mode)


class EnsureMode:
    def __init__(self, context, object: bpy.types.Object, target_mode: str):
        self._context = context
        self._object = object
        self._target_mode = target_mode

    def __enter__(self):
        cur_mode = self._object.mode
        if cur_mode != self._target_mode:
            self._prev_mode = cur_mode
            object_set_mode(self._context, self._object, self._target_mode)
        else:
            self._prev_mode = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._prev_mode is not None:
            object_set_mode(self._context, self._object, self._prev_mode)


def str_filter_alphanum(s: str, repl_c="_"):
    return "".join(
        c if "a" <= c <= "z" or "A" <= c <= "Z" or "0" <= c <= "9" else repl_c
        for c in s
    )


VertexIndex = int
TriIndices = tuple[VertexIndex, VertexIndex, VertexIndex]


def compute_n_cache_misses(faces_indices: list[TriIndices], cache_size: int):
    """Compute the amount of cache misses for some geometry and a given cache size."""

    cache = [None] * cache_size
    n_cache_misses = 0
    for tri in faces_indices:
        for v in tri:
            if v in cache:
                cache.remove(v)
            else:
                n_cache_misses += 1
            cache.insert(0, v)
            cache = cache[:cache_size]

    return n_cache_misses


class FacesIndicesOptimizer(abc.ABC):
    def __init__(
        self,
        faces_indices: list[TriIndices],
        *,
        cache_size: int,
        can_cyclic_permute_tri_vertices: bool,
        get_vertex_load_cost_distance: typing.Callable[
            [VertexIndex, VertexIndex], float
        ],
    ):
        """
        get_vertex_load_cost_distance: given the last loaded vertex (first arg), compute the additional cost due to a context switch for loading another vertex (second vertex)
        """
        self._original_faces_indices = faces_indices
        self._cache_size = cache_size
        self._can_cyclic_permute_tri_vertices = can_cyclic_permute_tri_vertices
        self._get_vertex_load_cost_distance = get_vertex_load_cost_distance

    _optimized_faces_indices: list[TriIndices]

    @abc.abstractmethod
    def optimize(self):
        """Implementation should compute and set self._optimized_faces_indices"""
        ...

    def get_optimized_faces_indices(self):
        return self._optimized_faces_indices


class FacesIndicesOptimizerNoop(FacesIndicesOptimizer):
    def optimize(self):
        self._optimized_faces_indices = self._original_faces_indices


class FacesIndicesOptimizerScoreBased(FacesIndicesOptimizer, abc.ABC):
    @abc.abstractmethod
    def _get_score_vertex(self, v: VertexIndex) -> float:
        ...

    def _get_score_triangle(
        self,
        get_vertex_score: typing.Callable[[VertexIndex], float],
        tri: TriIndices,
    ):
        return sum(map(get_vertex_score, tri))

    def _get_model_cache_size(self):
        return self._cache_size

    consider_vertex_load_cost = True

    def optimize(self):
        geo = self._original_faces_indices.copy()
        geo_opt = []
        cache_size = self._get_model_cache_size()
        cache: list[VertexIndex | None] = [None] * cache_size
        self._cache = cache

        # The amount of tris per vertex may be used in scoring the vertices.
        # It is a lot faster to maintain a dictionary and update it every iteration,
        # than recompute the values every time.
        n_tris_by_vertex: dict[VertexIndex, int] = dict()
        self._n_tris_by_vertex = n_tris_by_vertex
        for tri in geo:
            for v in tri:
                n_tris_by_vertex.setdefault(v, 0)
                n_tris_by_vertex[v] += 1

        @functools.cache
        def get_cyclic_permutations(tri: TriIndices):
            return (
                tri,
                (tri[1], tri[2], tri[0]),
                (tri[2], tri[0], tri[1]),
            )

        while geo:
            # reset the memoization every iteration
            get_vertex_score_cached = functools.cache(self._get_score_vertex)

            # find triangles with highest score
            max_score = -math.inf
            tris_with_max_score: list[tuple[int, TriIndices]] = []
            for i_tri, tri in enumerate(geo):
                tri_score = self._get_score_triangle(get_vertex_score_cached, tri)
                if tri_score > max_score:
                    max_score = tri_score
                    tris_with_max_score = []
                if tri_score == max_score:
                    tris_with_max_score.append((i_tri, tri))
            assert len(tris_with_max_score) >= 1

            if self.consider_vertex_load_cost:
                # TODO explain this
                # TODO need to also take into account permuting the tri indices to avoid evicting the tri's in-cache verts

                v_last_load = cache[0]

                def compute_additional_tri_cost_based_on_vertex_load_cost(tri):
                    tri_cost = 0
                    v_prev = tri[0] if v_last_load is None else v_last_load
                    for v_i in tri:
                        tri_cost += self._get_vertex_load_cost_distance(v_prev, v_i)
                        v_prev = v_i
                    return tri_cost

                min_tri_cost = math.inf
                i_tri_with_min_tri_cost = None
                tri_with_min_tri_cost = None
                for i_tri, tri in tris_with_max_score:
                    if self._can_cyclic_permute_tri_vertices:
                        tri = min(
                            get_cyclic_permutations(tri),
                            key=compute_additional_tri_cost_based_on_vertex_load_cost,
                        )
                    tri_cost = compute_additional_tri_cost_based_on_vertex_load_cost(
                        tri
                    )
                    if tri_cost < min_tri_cost:
                        min_tri_cost = tri_cost
                        i_tri_with_min_tri_cost = i_tri
                        tri_with_min_tri_cost = tri
                assert tri_with_min_tri_cost is not None
                i_tri, tri = i_tri_with_min_tri_cost, tri_with_min_tri_cost
            else:
                i_tri, tri = tris_with_max_score[0]

            geo.pop(i_tri)

            # update cache
            for v in tri:
                if v in cache:
                    cache.remove(v)
                cache.insert(0, v)
                cache = cache[:cache_size]

            # update remaining vertex usage
            for v in tri:
                n_tris_by_vertex[v] -= 1

            geo_opt.append(tri)

        assert all(n == 0 for n in n_tris_by_vertex.values())

        self._optimized_faces_indices = geo_opt


class FacesIndicesOptimizerForsyth(FacesIndicesOptimizerScoreBased):
    # consider_vertex_load_cost should be set to False to match Forsyth's simpler
    # algorithm (does not take the so-I-called "additional load cost" into account)
    # but it (for some reason) makes Forsyth's pecm better enough that its results
    # are used instead of the other though it may be better idk
    # TODO measure effect of taking consider_vertex_load_cost into account
    # consider_vertex_load_cost = False
    MaxSizeVertexCache = 1000

    def _get_model_cache_size(self):
        return self.MaxSizeVertexCache

    def Forsyth_FindVertexScore(self, num_active_tris, cache_position):
        # https://tomforsyth1000.github.io/papers/fast_vert_cache_opt.html

        FindVertexScore_CacheDecayPower = 1.5
        FindVertexScore_LastTriScore = 0.75
        FindVertexScore_ValenceBoostScale = 2.0
        FindVertexScore_ValenceBoostPower = 0.5

        MaxSizeVertexCache = self.MaxSizeVertexCache

        if num_active_tris == 0:
            return -1

        Score = 0.0

        if cache_position >= 0:
            if cache_position < 3:
                Score = FindVertexScore_LastTriScore
            else:
                assert cache_position < MaxSizeVertexCache
                Scaler = 1 / (MaxSizeVertexCache - 3)
                Score = 1 - (cache_position - 3) * Scaler
                Score = Score**FindVertexScore_CacheDecayPower

        ValenceBoost = num_active_tris**-FindVertexScore_ValenceBoostPower
        Score += FindVertexScore_ValenceBoostScale * ValenceBoost

        return Score

    def _get_score_vertex(self, v: VertexIndex):
        try:
            cache_position = self._cache.index(v)
        except ValueError:
            cache_position = -1
        return self.Forsyth_FindVertexScore(self._n_tris_by_vertex[v], cache_position)


class FacesIndicesOptimizerSimple(FacesIndicesOptimizerScoreBased):
    def _get_score_vertex(self, v: VertexIndex):
        n_tris = self._n_tris_by_vertex[v]
        if n_tris == 0:
            return -1
        """
        score: 1/n_tris + (cached?C:0)
        v0, v1, v2 in cache used by n_tris=inf
        v3, v4 in cache used by n_tris=1
        v5 not in cache used by n_tris=1
        score of tri1 v0,v1,v2: 3*(0+C)=3*C
        score of tri2 v3,v4,v5: 2*(1+C)+(1+0)=3+2*C
        tri1 is fully in cache so we want 3*C>3+2*C
        aka C>3

        note: in very limited testing, C=0.5 seemed fined and larger values didn't change anything

        indeed changing C from 0.5 to 4 didn't change the pecm at all (chance?)...
        """

        """
        score design:
        notation:
        v(n_tris,is_in_cache) for a vertex
        tri(v(...),...) for a face (order of vertices ignored)

        we want (selecting highest score(s))
        "all vertices cached" > "2/3 vertices cached" > "1/3 vertices cached" > "no vertices cached"
            tri(v(*,T),v(*,T),v(*,T)) > tri(v(*,T),v(*,T),v(*,F)) > tri(v(*,T),v(*,F),v(*,F)) > tri(v(*,F),v(*,F),v(*,F))
        prefer triangles with low n_tris of verts
            if a,b,c < d,e,f
            tri(v(a,T),v(b,T),v(c,F)) > tri(v(d,T),v(e,T),v(f,F))
            ...
        (todo)
        """

        return 1 / n_tris + (4 if v in self._cache else 0)


"""
a material is defined as
any kind of opengl draw parameter
that can only be changed outside a glBegin/glEnd pair
and that applies to a collection of faces (which end up in a glBegin/glEnd pair)

the material may assume the current state is already configured in some ways
(specific to whatever usage of the model is done)
but it should not assume anything from a previous material
(though the model writer may optimize things if possible)

that default current state is probably ideally taken as the opengl default state

typically the application is textures
but it could also be lighting properties or nearly anything (depth test, alpha test, alpha func...)
"""


class Vg64Material(abc.ABC):
    def __init__(self, indicative_name: str):
        self.indicative_name = indicative_name

    @abc.abstractmethod
    def write_apply(self, cw: CWriter):
        ...

    @abc.abstractmethod
    def write_revert(self, cw: CWriter):
        ...


class MaterialsCompound:
    def __init__(self, materials: frozenset[Vg64Material]):
        self.materials = materials


class MaterialTexture(Vg64Material):
    def __init__(self, indicative_name: str, texture_glname_variable: str):
        super().__init__(indicative_name)
        self.texture_glname_variable = texture_glname_variable

    def write_apply(self, cw: CWriter):
        cw.writeln("glEnable(GL_TEXTURE_2D);")
        cw.writeln(f"glBindTexture(GL_TEXTURE_2D, {self.texture_glname_variable});")

    def write_revert(self, cw: CWriter):
        cw.writeln("glDisable(GL_TEXTURE_2D);")


Vec2f = tuple[float, float]
Vec3f = tuple[float, float, float]
Vec4f = tuple[float, float, float, float]


@dataclasses.dataclass(frozen=True)
class Vg64Bone:
    name: str


# a mix of blender's loop (corner) and vertex
@dataclasses.dataclass(frozen=True)
class Vg64Vertex:
    pos: Vec3f
    normal: Vec3f = None
    color: Vec4f = None
    uv: Vec2f = None
    rigged_to: Vg64Bone = None


@dataclasses.dataclass(frozen=True)
class Vg64Triangle:
    vertices: tuple[Vg64Vertex, Vg64Vertex, Vg64Vertex]


@dataclasses.dataclass(frozen=True)
class Vg64Mesh:
    faces_by_mc: dict[MaterialsCompound, list[Vg64Triangle]]


@dataclasses.dataclass
class ExportOptions:
    normals: bool = False
    vertex_colored: bool = False
    vertex_colored_skip_alpha: bool = False
    textured: bool = False
    rigged: bool = False
    cache_size: int = 32
    can_cyclic_permute_tri_vertices: bool = True
    """Set to False if using flat color shading and not all vertices have the same color
    (afaik the only reason to set this to False)"""
    # TODO not that False makes the result predictable, since I don't think you can
    # control which vertex/loop of a face is first through blender...


@dataclasses.dataclass
class Vg64MeshCArgument:
    type_c: str
    name_c: str
    comments: str


class Vg64MeshExporter:
    def __init__(
        self,
        cwriter: CWriter,
        object: bpy.types.Object,
        options: ExportOptions,
        context: bpy.types.Context,
    ):
        self.cwriter = cwriter
        self.object = object
        self.options = options
        self.context = context

        self.arguments: list[Vg64MeshCArgument] = []

        self.materials_compound_cache: (
            dict[frozenset[Vg64Material], MaterialsCompound]
        ) = dict()
        self.materials_from_blmaterial_cache: (
            dict[bpy.types.Material, list[Vg64Material]]
        ) = dict()
        self.materials_from_blimage_cache: (
            dict[bpy.types.Image, MaterialTexture]
        ) = dict()

    def _get_materials_from_blmaterial(self, blmaterial: bpy.types.Material):
        materials = self.materials_from_blmaterial_cache.get(blmaterial)
        if materials is not None:
            return materials

        materials = []

        # TODO more material stuff (alpha test func, blend func, lighting, multitexture)

        if blmaterial.use_nodes and blmaterial.node_tree:
            image_texture_nodes = [
                n
                for n in blmaterial.node_tree.nodes
                if n.bl_idname == "ShaderNodeTexImage"
            ]
            if image_texture_nodes:
                image = next(
                    (n.image for n in image_texture_nodes if n.image is not None),
                    None,
                )
                if image:
                    assert isinstance(image, bpy.types.Image)
                    tex_material = self.materials_from_blimage_cache.get(image)
                    if tex_material is None:
                        texture_glname_variable = "tex_" + str_filter_alphanum(
                            image.name
                        )
                        self.arguments.append(
                            Vg64MeshCArgument(
                                "GLuint",
                                texture_glname_variable,
                                f"image {image.name}",
                            )
                        )
                        tex_material = MaterialTexture(
                            f"MaterialTexture {image.name}",
                            texture_glname_variable,
                        )
                    materials.append(tex_material)

        self.materials_from_blmaterial_cache[blmaterial] = materials
        return materials

    def _get_materials_compound(
        self,
        tri: bpy.types.MeshLoopTriangle,
        o_eval: bpy.types.Object,
    ):
        materials: list[Vg64Material] = []
        if 0 <= tri.material_index < len(o_eval.material_slots):
            blmaterial_slot = o_eval.material_slots[tri.material_index]
            if blmaterial_slot is not None:
                blmaterial = blmaterial_slot.material
                if blmaterial is not None:
                    materials.extend(self._get_materials_from_blmaterial(blmaterial))

        materials_set = frozenset(materials)

        mc = self.materials_compound_cache.get(materials_set)
        if mc is None:
            mc = MaterialsCompound(materials_set)
            self.materials_compound_cache[materials_set] = mc

        return mc

    def _get_mesh_data(self, o_eval: bpy.types.Object):
        options = self.options

        mesh_eval: bpy.types.Mesh = o_eval.data

        mesh_eval.calc_normals_split()  # for loop.normal

        if options.vertex_colored:
            color_attribute = mesh_eval.color_attributes.active_color
            assert color_attribute is not None
            assert color_attribute.data_type in {
                "BYTE_COLOR",
                "FLOAT_COLOR",
            }
            assert color_attribute.domain in {"POINT", "CORNER"}
            if color_attribute.data_type == "BYTE_COLOR":
                assert isinstance(color_attribute, bpy.types.ByteColorAttribute)

                def get_color_from_color_attribute_value(
                    v: bpy.types.ByteColorAttributeValue,
                ):
                    # a 4-floats tuple
                    return v.color

            elif color_attribute.data_type == "FLOAT_COLOR":
                assert isinstance(color_attribute, bpy.types.FloatColorAttribute)

                def get_color_from_color_attribute_value(
                    v: bpy.types.FloatColorAttributeValue,
                ):
                    # a 4-floats tuple
                    return v.color

            else:
                assert False

        if options.textured:
            uvs = mesh_eval.uv_layers.active.uv

        if options.rigged:
            vg64bone_by_group_index: dict[int, Vg64Bone] = dict()

        faces_by_mc = dict()
        for tri in mesh_eval.loop_triangles:
            vg64vertices = []
            for loop_index in tri.loops:
                loop = mesh_eval.loops[loop_index]
                if options.normals:
                    normal = loop.normal

                if options.vertex_colored:
                    if color_attribute.domain == "CORNER":
                        color = get_color_from_color_attribute_value(
                            color_attribute.data[loop_index]
                        )

                if options.textured:
                    uv = uvs[loop_index].vector

                vertex_index = loop.vertex_index
                v = mesh_eval.vertices[vertex_index]
                if options.vertex_colored:
                    if color_attribute.domain == "POINT":
                        color = get_color_from_color_attribute_value(
                            color_attribute.data[vertex_index]
                        )

                co = v.co

                if options.rigged:
                    # TODO assert single weight of 1
                    vge = max(
                        v.groups,
                        key=lambda vge: vge.weight,
                        default=None,
                    )
                    rigged_to = vg64bone_by_group_index.get(vge.group)
                    if rigged_to is None:
                        rigged_to = Vg64Bone(self.object.vertex_groups[vge.group].name)
                        vg64bone_by_group_index[vge.group] = rigged_to

                vg64v = Vg64Vertex(
                    pos=tuple(co),
                    normal=tuple(normal) if options.normals else None,
                    color=tuple(color) if options.vertex_colored else None,
                    uv=tuple(uv) if options.textured else None,
                    rigged_to=rigged_to if options.rigged else None,
                )
                vg64vertices.append(vg64v)

            assert len(vg64vertices) == 3
            vg64tri = Vg64Triangle(tuple(vg64vertices))

            mc = self._get_materials_compound(tri, o_eval)

            faces_by_mc.setdefault(mc, []).append(vg64tri)

        return Vg64Mesh(faces_by_mc)

    def _write_mesh_vertices_beginendpair(self, faces: list[Vg64Triangle]):
        cw = self.cwriter
        options = self.options

        cw.writeln("glBegin(GL_TRIANGLES);")
        cw.writeln("")

        for tri in faces:
            for v in tri.vertices:
                if options.normals:
                    cw.writeln(
                        f"glNormal3f({v.normal[0]}, {v.normal[1]}, {v.normal[2]});"
                    )

                if options.vertex_colored:
                    if options.vertex_colored_skip_alpha:
                        cw.writeln(
                            f"glColor3f({v.color[0]}, {v.color[1]}, {v.color[2]});"
                        )
                    else:
                        cw.writeln(
                            f"glColor4f({v.color[0]}, {v.color[1]}, {v.color[2]}, {v.color[3]});"
                        )

                if options.textured:
                    cw.writeln(f"glTexCoord2f({v.uv[0]}, {v.uv[1]});")

                if options.rigged:
                    matrix_palette_index = self.matrix_palette_indices.get(v.rigged_to)
                    cw.writeln(
                        "glMatrixIndexubvARB(1, (GLubyte[]){"
                        + str(matrix_palette_index)
                        + "});"
                    )

                with cw.indented():
                    cw.writeln(f"glVertex3f({v.pos[0]}, {v.pos[1]}, {v.pos[2]});")
            cw.writeln("")

        cw.writeln("glEnd();")

    def _make_vertices_indices(self, faces: list[Vg64Triangle]):
        """Assign an index from 0 to each unique vertex,
        and return the list of vertices in the corresponding order,
        and the list of indices for each face.
        (this handles deduplicating identical vertices)"""

        vertices_ordered: list[Vg64Vertex] = []
        vertices_indices: dict[Vg64Vertex, VertexIndex] = dict()
        faces_indices: list[TriIndices] = []

        for tri in faces:
            tri_indices = []
            for v in tri.vertices:
                v_i = vertices_indices.get(v)
                if v_i is None:
                    v_i = len(vertices_ordered)
                    vertices_ordered.append(v)
                    vertices_indices[v] = v_i
                tri_indices.append(v_i)
            faces_indices.append(tuple(tri_indices))

        return vertices_ordered, faces_indices

    def _optimize_faces_indices(self, faces: list[Vg64Triangle]):
        assert len(faces) != 0

        options = self.options

        vertices_ordered, faces_indices = self._make_vertices_indices(faces)

        faces_indices_optimizers: list[type[FacesIndicesOptimizer]] = [
            FacesIndicesOptimizerNoop,
            FacesIndicesOptimizerForsyth,
            FacesIndicesOptimizerSimple,
        ]

        def get_vertex_load_cost_distance(v_ctx_i: VertexIndex, v_i: VertexIndex):
            if not options.rigged:
                return 0
            v_ctx = vertices_ordered[v_ctx_i]
            v = vertices_ordered[v_i]
            if (
                self.matrix_palette_indices[v_ctx.rigged_to]
                == self.matrix_palette_indices[v.rigged_to]
            ):
                return 0
            else:
                return 1

        # TODO maybe consider those if I can find a good way
        # to compare results without in-game testing
        # FacesIndicesOptimizerForsyth.consider_vertex_load_cost = False
        # FacesIndicesOptimizerSimple.consider_vertex_load_cost = False

        min_n_cache_misses = math.inf
        faces_indices_with_min_n_cache_misses = None
        for fio_type in faces_indices_optimizers:
            fio = fio_type(
                faces_indices,
                cache_size=options.cache_size,
                can_cyclic_permute_tri_vertices=options.can_cyclic_permute_tri_vertices,
                get_vertex_load_cost_distance=get_vertex_load_cost_distance,
            )
            fio.optimize()
            optimized_faces_indices = fio.get_optimized_faces_indices()
            n_cache_misses = compute_n_cache_misses(
                optimized_faces_indices, options.cache_size
            )
            if n_cache_misses < min_n_cache_misses:
                min_n_cache_misses = n_cache_misses
                faces_indices_with_min_n_cache_misses = optimized_faces_indices

            n_tris = len(faces_indices)
            n_unique_vertices = len({v for tri in faces_indices for v in tri})
            pecm = (n_cache_misses - n_unique_vertices) / (
                3 * n_tris - n_unique_vertices
            )
            v_prev_i = optimized_faces_indices[0][0]
            total_vlcd = 0
            for v_i in [v for tri in optimized_faces_indices for v in tri][1:]:
                total_vlcd += get_vertex_load_cost_distance(v_prev_i, v_i)
                v_prev_i = v_i
            print(fio_type.__name__, n_cache_misses, pecm, total_vlcd)
            self.cwriter.writeln(
                f"// {fio_type.__name__} n_cm={n_cache_misses} pecm={pecm} tot_vlcd={total_vlcd}"
            )

        assert faces_indices_with_min_n_cache_misses is not None

        return vertices_ordered, faces_indices_with_min_n_cache_misses

    def _write_mesh_vertices_vertexarray(
        self,
        vertices_ordered: list[Vg64Vertex],
        faces_indices: list[tuple[int, int, int]],
    ):
        cw = self.cwriter
        options = self.options

        vertices_data_name = "vertices"

        cw.writeln("struct {")
        with cw.indented():
            cw.writeln("GLfloat pos[3];")
            if options.normals:
                cw.writeln("GLfloat normal[3];")
            if options.vertex_colored:
                if options.vertex_colored_skip_alpha:
                    cw.writeln("GLfloat color[3];")
                else:
                    cw.writeln("GLfloat color[4];")
            if options.textured:
                cw.writeln("GLfloat tc[2];")
            if options.rigged:
                cw.writeln("GLubyte matrix_index;")
        cw.writeln("} " + vertices_data_name + "[] = {")

        with cw.indented():
            for v_i, v in enumerate(vertices_ordered):
                fields_vals = []

                fields_vals.append("{" f"{v.pos[0]}, {v.pos[1]}, {v.pos[2]}" "}")

                if options.normals:
                    fields_vals.append(
                        "{" f"{v.normal[0]}, {v.normal[1]}, {v.normal[2]}" "}"
                    )

                if options.vertex_colored:
                    if options.vertex_colored_skip_alpha:
                        fields_vals.append(
                            "{" f"{v.color[0]}, {v.color[1]}, {v.color[2]}" "}"
                        )
                    else:
                        fields_vals.append(
                            "{"
                            f"{v.color[0]}, {v.color[1]}, {v.color[2]}, {v.color[3]}"
                            "}"
                        )

                if options.textured:
                    fields_vals.append("{" f"{v.uv[0]}, {v.uv[1]}" "}")

                if options.rigged:
                    matrix_palette_index = self.matrix_palette_indices.get(v.rigged_to)
                    fields_vals.append(str(matrix_palette_index))

                cw.write(f"/* {v_i} */ ")
                cw.write("{ ")
                cw.write(", ".join(fields_vals))
                cw.write(" },\n")

        cw.writeln("};")

        n_vertices = len(vertices_ordered)
        for n_vertices_max, type_c, type_glenum in (
            (2**8, "GLubyte", "GL_UNSIGNED_BYTE"),
            (2**16, "GLushort", "GL_UNSIGNED_SHORT"),
            (2**32, "GLuint", "GL_UNSIGNED_INT"),
        ):
            if n_vertices <= n_vertices_max:
                indices_data_type_c = type_c
                indices_data_type_glenum = type_glenum
                break
        else:
            assert False, f"too many vertices {n_vertices}"

        indices_data_name = "indices"

        cw.writeln(f"{indices_data_type_c} {indices_data_name}[] = " "{")
        with cw.indented():
            for tri_indices in faces_indices:
                cw.writeln(
                    " ".join(f"{v_i}," for v_i in tri_indices)
                    + (
                        (
                            " // "
                            + ",".join(
                                map(
                                    lambda v_i: str(
                                        self.matrix_palette_indices[
                                            vertices_ordered[v_i].rigged_to
                                        ]
                                    ),
                                    tri_indices,
                                )
                            )
                        )
                        if options.rigged
                        else ""
                    )
                )
        cw.writeln("};")

        enabled_arrays = []
        enabled_arrays.append("GL_VERTEX_ARRAY")
        if options.normals:
            enabled_arrays.append("GL_NORMAL_ARRAY")
        if options.vertex_colored:
            enabled_arrays.append("GL_COLOR_ARRAY")
        if options.textured:
            enabled_arrays.append("GL_TEXTURE_COORD_ARRAY")
        if options.rigged:
            enabled_arrays.append("GL_MATRIX_INDEX_ARRAY_ARB")

        cw.writeln("")
        for array in enabled_arrays:
            cw.writeln(f"glEnableClientState({array});")

        cw.writeln("")
        cw.writeln(
            "glVertexPointer(3, GL_FLOAT, "
            f"sizeof({vertices_data_name}[0]), "
            f"&{vertices_data_name}[0].pos);"
        )
        if options.normals:
            cw.writeln(
                "glNormalPointer(GL_FLOAT, "
                f"sizeof({vertices_data_name}[0]), "
                f"&{vertices_data_name}[0].normal);"
            )
        if options.vertex_colored:
            if options.vertex_colored_skip_alpha:
                cw.writeln(
                    "glColorPointer(3, GL_FLOAT, "
                    f"sizeof({vertices_data_name}[0]), "
                    f"&{vertices_data_name}[0].color);"
                )
            else:
                cw.writeln(
                    "glColorPointer(4, GL_FLOAT, "
                    f"sizeof({vertices_data_name}[0]), "
                    f"&{vertices_data_name}[0].color);"
                )
        if options.textured:
            cw.writeln(
                "glTexCoordPointer(2, GL_FLOAT, "
                f"sizeof({vertices_data_name}[0]), "
                f"&{vertices_data_name}[0].tc);"
            )
        if options.rigged:
            cw.writeln(
                "glMatrixIndexPointerARB(1, GL_UNSIGNED_BYTE, "
                f"sizeof({vertices_data_name}[0]), "
                f"&{vertices_data_name}[0].matrix_index);"
            )

        cw.writeln("")
        cw.writeln(
            "glDrawElements(GL_TRIANGLES, "
            f"sizeof({indices_data_name}) / sizeof({indices_data_name}[0]), "
            f"{indices_data_type_glenum}, {indices_data_name});"
        )

        cw.writeln("")
        for array in enabled_arrays:
            cw.writeln(f"glDisableClientState({array});")

    def write(self):
        cw = self.cwriter
        object = self.object
        options = self.options
        context = self.context

        with EnsureMode(context, object, "OBJECT"):
            depsgraph = context.evaluated_depsgraph_get()
            o_eval = object.evaluated_get(depsgraph)
            vg64mesh = self._get_mesh_data(o_eval)

        if options.rigged:
            self.matrix_palette_indices: dict[Vg64Bone, int] = {
                b: i
                for i, b in enumerate(
                    # sorting makes the index ordering consistent
                    # (it's not otherwise required for anything)
                    sorted(
                        {
                            v.rigged_to
                            for faces in vg64mesh.faces_by_mc.values()
                            for tri in faces
                            for v in tri.vertices
                        },
                        key=lambda b: b.name,
                    )
                )
            }
        c_name = str_filter_alphanum(object.name)
        cw.writeln(f"void model_{c_name}(")
        with cw.indented():
            if self.arguments:
                cw.writeln(
                    ",\n".join(
                        f"/* {arg.comments} */\n{arg.type_c} {arg.name_c}"
                        for arg in self.arguments
                    )
                )
            else:
                cw.writeln("void")
        cw.writeln(")")
        cw.writeln("{")

        with cw.indented():
            # TODO sort by mc (eg reduce costly context switching like texture change)
            """
            eg:
            texA + lightA
            texB + lightA
            texA + lightB
            would be faster ordered like:
            texA + lightA
            texA + lightB
            texB + lightA
            and omitting the material revert/apply in between the two texA
            """
            # this probably calls for a way to compare material enable/disable costs

            active_materials: frozenset[Vg64Material] = frozenset()

            for mc, faces in vg64mesh.faces_by_mc.items():
                materials_to_revert = active_materials - mc.materials
                if materials_to_revert:
                    cw.writeln("// Material revert")
                    for m in materials_to_revert:
                        cw.writeln(f"// {m.indicative_name}")
                        with cw.indented():
                            m.write_revert(cw)

                materials_to_apply = mc.materials - active_materials
                if materials_to_apply:
                    cw.writeln("// Material apply")
                    for m in mc.materials:
                        cw.writeln(f"// {m.indicative_name}")
                        with cw.indented():
                            m.write_apply(cw)

                materials_kept_applied = active_materials & mc.materials
                if materials_kept_applied:
                    cw.writeln("// Materials kept:")
                    with cw.indented():
                        for m in materials_kept_applied:
                            cw.writeln(f"// {m.indicative_name}")

                active_materials = mc.materials

                vertices_ordered, faces_indices = self._optimize_faces_indices(faces)

                # self._write_mesh_vertices_beginendpair(faces)
                self._write_mesh_vertices_vertexarray(vertices_ordered, faces_indices)

            if active_materials:
                cw.writeln("// Final material revert")
                for m in active_materials:
                    cw.writeln(f"// {m.indicative_name}")
                    with cw.indented():
                        m.write_revert(cw)

        cw.writeln("}")

        if options.rigged:
            cw.writeln("/*")
            for vg64bone, matrix_palette_index in self.matrix_palette_indices.items():
                cw.writeln(f"{matrix_palette_index} {vg64bone.name}")
            cw.writeln("*/")


class Vg64Exporter:
    def __init__(
        self,
        f,  # text file object
    ):
        self.cwriter = CWriter(f.write)

    def write_preamble(self):
        cw = self.cwriter
        cw.writeln("#include <GL/gl.h>")

    def write_object(
        self,
        object: bpy.types.Object,
        options: ExportOptions,
        context=None,
    ):
        if context is None:
            context = bpy.context
        assert object.type == "MESH"

        me = Vg64MeshExporter(self.cwriter, object, options, context)
        me.write()


def export(object, f, options, context):
    e = Vg64Exporter(f)
    e.write_preamble()
    e.write_object(object, options, context)


class ExportSomeData(bpy.types.Operator):
    bl_idname = "export.some_data"
    bl_label = "Export Some Data"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    @classmethod
    def poll(cls, context):
        o = context.object
        return o is not None and o.type == "MESH"

    def execute(self, context):
        with Path(self.filepath).open("w") as f:
            export(context.object, f, ExportOptions(), context)
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


def menu_func(self, context):
    self.layout.operator_context = "INVOKE_DEFAULT"
    self.layout.operator(ExportSomeData.bl_idname, text="Text Export Operator")


def register():
    bpy.utils.register_class(ExportSomeData)
    bpy.types.TOPBAR_MT_file_export.append(menu_func)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_func)
    bpy.utils.unregister_class(ExportSomeData)


def main():
    def write(s):
        print(s, end="")

    cw = CWriter(write)
    cw.writeln("hi()")
    with cw.indented():
        cw.writeln("...")


if __name__ == "__main__":
    main()
