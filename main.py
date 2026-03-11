import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class WorldOrder:
    """Bounding box and collision container."""
    def __init__(self, bounds=(100, 100, 100)):
        self.bounds = np.array(bounds)
        self.obstacles = []      # List of (min_corner, max_corner)
        self.baked_paths = []

    def add_obstacle(self, center, size):
        center, size = np.array(center), np.array(size)
        min_c = center - (size / 2)
        max_c = center + (size / 2)
        self.obstacles.append((min_c, max_c))

    def _get_cube_edges(self, min_c, max_c):
        """Helper to calculate the 12 lines of a box for plotting."""
        # The 8 vertices of a box
        z = np.array([
            [min_c[0], min_c[1], min_c[2]],
            [max_c[0], min_c[1], min_c[2]],
            [max_c[0], max_c[1], min_c[2]],
            [min_c[0], max_c[1], min_c[2]],
            [min_c[0], min_c[1], max_c[2]],
            [max_c[0], min_c[1], max_c[2]],
            [max_c[0], max_c[1], max_c[2]],
            [min_c[0], max_c[1], max_c[2]],
        ])

        # List of vertex pairs to connect for edges
        edges = [
            [z[0],z[1]], [z[1],z[2]], [z[2],z[3]], [z[3],z[0]],
            [z[4],z[5]], [z[5],z[6]], [z[6],z[7]], [z[7],z[4]],
            [z[0],z[4]], [z[1],z[5]], [z[2],z[6]], [z[3],z[7]],
        ]
        return edges

class Collector:
    """Manifold that generates specific entry/exit ports."""

    def __init__(self, pos=(0,0,0), direction=(0,0,1), num_ports=4, radius=5.0, length=10.0):
        # pos: 3D coordinates (base center)
        # direction: exit direction (end_dir)
        # num_ports: total number of ports (n)
        # radius: radial distance from center axis to port center (r)
        # length: distance from base to end port (l)

        self.pos = np.array(pos)
        self.direction = self._normalize(np.array(direction))
        self.n = num_ports
        self.r = radius
        self.l = length

        # Generate port positions
        self.ports = self._generate_ports()

    def _normalize(self, v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def _get_orthogonal_basis(self):
        """Creates vectors u and v perpendicular to the manifold direction."""
        # Pick an arbitrary vector not parallel to direction
        pick = np.array([1, 0, 0]) if abs(self.direction[0]) < 0.9 else np.array([0, 1, 0])
        u = self._normalize(np.cross(self.direction, pick))
        v = np.cross(self.direction, u)
        return u, v

    def _generate_ports(self):
        """Generates port positions spaced by 360/n degrees."""
        ports = []
        u, v = self._get_orthogonal_basis()

        angle_step = 2 * np.pi / self.n

        for i in range(self.n):
            theta = i * angle_step
            # Calculate position on the radial ring
            radial_offset = self.r * (np.cos(theta) * u + np.sin(theta) * v)
            port_pos = self.pos + radial_offset

            ports.append({
                'id': i,
                'position': port_pos,
                # 'normal': self._normalize(radial_offset),
                'normal': self.direction,
            })
        return ports

    def get_end_point(self):
        """Returns the 'end port' location."""
        return self.pos + (self.direction * self.l)

class SplinePath:
    """Data container for a single routed pipe."""
    def __init__(self, start_pos, end_pos, start_dir, end_dir, radius=1.0):
        self.start_pos = np.array(start_pos)
        self.start_dir = self._normalize(np.array(start_dir))
        self.end_pos = np.array(end_pos)
        self.end_dir = self._normalize(np.array(end_dir))

        self.radius = radius
        self.coords = [self.start_pos]  # Initialize with start point
        self.length = 0.0
        self.complete = False

    def _normalize(self, v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def update_path(self, new_coords):
        """Updates the path points and recalculates total length."""
        self.coords = np.array(new_coords)
        # Calculate length by summing distances between consecutive points
        diffs = np.diff(self.coords, axis=0)
        self.length = np.sum(np.sqrt((diffs**2).sum(axis=1)))

        # Check if the last point is close enough to end_pos to call it 'complete'
        if np.linalg.norm(self.coords[-1] - self.end_pos) < 1e-3:
            self.complete = True

    def get_bend_angles(self):
        """Returns the angles between segments to check if bend < 90 deg."""
        vectors = np.diff(self.coords, axis=0)
        norms = np.linalg.norm(vectors, axis=1)
        angles = []
        for i in range(len(vectors) - 1):
            v1 = vectors[i] / norms[i]
            v2 = vectors[i+1] / norms[i+1]
            angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
            angles.append(angle)
        return angles

start_pos = np.array([
    [7, 0, 7],
    [7, 5, 7],
    [7, 10, 7],
    [7, 15, 7],
])
my_world = WorldOrder(bounds=(20, 20, 20))
coll = Collector(
    pos=(0, 0, 0),
    direction=(0, 0, -1),
    num_ports=4,
    radius=4,
    length=12
)
