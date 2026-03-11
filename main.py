import numpy as np
import matplotlib.pyplot as plt

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
    """Data container for a single routed pipe using Cubic Bézier interpolation."""
    def __init__(self, start_pos, end_pos, start_dir, end_dir, radius=1.0):
        self.start_pos = np.array(start_pos)
        self.start_dir = self._normalize(np.array(start_dir))
        self.end_pos = np.array(end_pos)
        self.end_dir = self._normalize(np.array(end_dir))

        self.radius = radius
        self.coords = []
        self.length = 0.0
        self.complete = False

    def _normalize(self, v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def generate_bezier(self, multiplier, num_points=50):
        """
        Calculates a smooth curve using start/end points and tangents.
        P1 and P2 are determined by the 'multiplier' (stiffness).
        """
        # Control Point 0: The start
        p0 = self.start_pos
        # Control Point 1: Extruded from start along start_dir
        p1 = self.start_pos + (self.start_dir * multiplier)
        # Control Point 2: Backed off from end along end_dir
        p2 = self.end_pos - (self.end_dir * multiplier)
        # Control Point 3: The actual port
        p3 = self.end_pos

        # Generate t-values from 0 to 1
        t = np.linspace(0, 1, num_points)[:, None]

        # Cubic Bézier Formula: (1-t)^3*P0 + 3(1-t)^2*t*P1 + 3(1-t)t^2*P2 + t^3*P3
        bezier_coords = (1 - t)**3 * p0 + \
                        3 * (1 - t)**2 * t * p1 + \
                        3 * (1 - t) * t**2 * p2 + \
                        t**3 * p3

        self.update_path(bezier_coords)

    def update_path(self, new_coords):
        """Updates coordinates and calculates the actual arc length."""
        self.coords = np.array(new_coords)
        # Sum of Euclidean distances between each sampled point
        diffs = np.diff(self.coords, axis=0)
        self.length = np.sum(np.sqrt((diffs**2).sum(axis=1)))

        if np.linalg.norm(self.coords[-1] - self.end_pos) < 0.1:
            self.complete = True

    def get_max_bend(self):
        """Calculates the sharpest angle in the path to verify < 90 deg."""
        if len(self.coords) < 3: return 0.0

        vectors = np.diff(self.coords, axis=0)
        norms = np.linalg.norm(vectors, axis=1)

        max_angle = 0.0
        for i in range(len(vectors) - 1):
            # Cosine similarity between adjacent segments
            v1 = vectors[i] / norms[i]
            v2 = vectors[i+1] / norms[i+1]
            dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
            angle = np.degrees(np.acos(dot))
            if angle > max_angle:
                max_angle = angle
        return max_angle

class RouterManager:
    def __init__(self, world, collector, starts):
        self.world = world
        self.collector = collector
        self.starts = np.array(starts)
        self.paths = []
        self.reference_length = 0.0

    def _get_furthest_pairing(self):
        """Rank starts to ports and find the global furthest for the Master Path."""
        max_dist = -1
        best_pair = (None, None)
        port_positions = [p['position'] for p in self.collector.ports]

        for s_idx, s_pos in enumerate(self.starts):
            for p_idx, p_pos in enumerate(port_positions):
                dist = np.linalg.norm(s_pos - p_pos)
                if dist > max_dist:
                    max_dist = dist
                    best_pair = (s_idx, p_idx)
        return best_pair

    def run_routing(self, base_multiplier=2.0):
        # 1. Identify the 'Master' (furthest) route
        start_idx, port_idx = self._get_furthest_pairing()

        # 2. Generate Master Path with base stiffness
        master_path = self._create_curved_path(start_idx, port_idx, multiplier=base_multiplier)
        self.reference_length = master_path.length
        self.paths.append(master_path)

        print(f"Master Route: Start {start_idx} -> Port {port_idx}")
        print(f"Reference Length Set: {self.reference_length:.2f}")

        # 3. Route the rest and match length by increasing multiplier
        remaining_starts = [i for i in range(len(self.starts)) if i != start_idx]
        remaining_ports = [i for i in range(len(self.collector.ports)) if i != port_idx]

        for s_idx, p_idx in zip(remaining_starts, remaining_ports):
            # Start with base stiffness
            mult = base_multiplier
            path = self._create_curved_path(s_idx, p_idx, multiplier=mult)

            # Iterative Multiplier adjustment to match length
            # We increase 'stiffness' to force a wider, longer curve
            while path.length < self.reference_length and mult < 15.0:
                mult += 0.5
                path = self._create_curved_path(s_idx, p_idx, multiplier=mult)

            self.paths.append(path)
            print(f"Matched Path: Start {s_idx} | Multiplier: {mult:.1f} | Length: {path.length:.2f}")

    def _create_curved_path(self, s_idx, p_idx, multiplier):
        start_p = self.starts[s_idx]
        port = self.collector.ports[p_idx]

        # 1. Start Direction: Shoot out along +X away from the wall
        s_dir = np.array([1, 0, 0])

        # 2. End Direction: Entry into the port
        # We use the port's normal (collector's axis)
        e_dir = port['normal']

        # 3. Define Path Object
        path = SplinePath(start_p, port['position'], s_dir, e_dir)

        # 4. FIX: Call 'generate_bezier' (the name used in the SplinePath class)
        # We don't need to pass p0-p3 manually anymore; the method handles it
        # using the start/end/dir info already stored in the object.
        path.generate_bezier(multiplier=multiplier, num_points=50)

        return path

    def plot_all(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot Collector
        ports = np.array([p['position'] for p in self.collector.ports])
        ax.scatter(ports[:,0], ports[:,1], ports[:,2], c='red', label='Collector Ports')

        # Plot Paths
        for i, p in enumerate(self.paths):
            pts = np.array(p.coords)
            ax.plot(pts[:,0], pts[:,1], pts[:,2], label=f'Path {i} (L={p.length:.1f})')

        ax.set_xlim(0, self.world.bounds[0])
        ax.set_ylim(0, self.world.bounds[1])
        ax.set_zlim(0, self.world.bounds[2])
        ax.legend()
        plt.show()

# --- Updated Initialization ---
start_pos = np.array([
    [0, 0, 7],
    [0, 5, 7],
    [0, 10, 7],
    [0, 15, 7],
])

my_world = WorldOrder(bounds=(20, 20, 20))
coll = Collector(
    pos=(5, 20, 0),
    direction=(0, 1, 0), # Pointing down
    num_ports=4,
    radius=2,
    length=5,
)

# --- Execution ---
manager = RouterManager(my_world, coll, start_pos)
manager.run_routing(base_multiplier=1.5) # Initial 'straight' lead-in
manager.plot_all()