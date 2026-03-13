import numpy as np
import matplotlib.pyplot as plt

class WorldOrder:
    """Bounding box and collision container."""
    def __init__(self, bounds=(25, 25, 25)):
        self.bounds = np.array(bounds) #

    def is_within_bounds(self, point):
        """Step 4-4: Boundary check logic."""
        # Returns True only if the point is inside the [0, bounds] cube
        return np.all(point >= 0) and np.all(point <= self.bounds)

class Collector:
    """Manifold that generates specific entry/exit ports based on user input."""
    def __init__(self, pos=(10, 20, 5), direction=(0, 1, 0), num_ports=4, radius=3.0):
        self.pos = np.array(pos) #
        self.direction = self._normalize(np.array(direction)) #
        self.n = num_ports #
        self.r = radius #
        self.ports = self._generate_ports() #

    def _normalize(self, v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def _get_orthogonal_basis(self):
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
            radial_offset = self.r * (np.cos(theta) * u + np.sin(theta) * v)
            # Normal is the direction the port faces (direction of the collector)
            ports.append({'id': i, 'position': self.pos + radial_offset, 'normal': self.direction})
        return ports

class SplinePath:
    """Data container for a routed pipe with length tracking."""
    def __init__(self, start_pos, end_pos):
        self.coords = []
        self.length = 0.0

    def update_path(self, new_coords):
        self.coords = np.array(new_coords)
        # Calculate Euclidean length for scavenging/backpressure feedback
        diffs = np.diff(self.coords, axis=0)
        self.length = np.sum(np.sqrt((diffs**2).sum(axis=1)))

class RouterManager:
    def __init__(self, world, collector, starts, pairings):
        self.world = world
        self.collector = collector
        self.starts = np.array(starts)
        self.pairings = pairings  # Manual start-port pairing
        self.paths = {}
        self.ref_len = 0.0

    def generate_path(self, pair_idx, stub_len, bias):
        """Step 4 & 5: Generates path with rigid stubs and a smooth bridge."""
        s_idx, p_idx = self.pairings[pair_idx]
        start_p = self.starts[s_idx]
        port = self.collector.ports[p_idx]

        # User-defined Start Direction
        s_dir = np.array([1, 0, 0])
        e_dir = port['normal']

        # 1. RIGID STUBS: Ensure the pipe leaves/enters straight
        p_stub_out = start_p + (s_dir * stub_len)
        p_stub_in = port['position'] - (e_dir * stub_len)

        # 2. BRIDGE: Cubic Bezier between stubs
        dist = np.linalg.norm(p_stub_in - p_stub_out)
        mult = dist * bias
        t = np.linspace(0, 1, 30)[:, None]
        cp0, cp3 = p_stub_out, p_stub_in
        cp1, cp2 = cp0 + (s_dir * mult), cp3 - (e_dir * mult)

        # Cubic Bezier Formula
        bridge = (1-t)**3*cp0 + 3*(1-t)**2*t*cp1 + 3*(1-t)*t**2*cp2 + t**3*cp3

        # 3. ASSEMBLY
        full_coords = np.vstack([start_p, p_stub_out, bridge, p_stub_in, port['position']])
        path = SplinePath(start_p, port['position'])
        path.update_path(full_coords)

        # 4. BBOX CHECK
        for pt in path.coords:
            if not self.world.is_within_bounds(pt):
                print(f"--- WARNING: Path {pair_idx} out of bounds at {pt} ---")
                break

        return path

    def plot(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for i, path in self.paths.items():
            pts = path.coords
            ax.plot(pts[:,0], pts[:,1], pts[:,2], label=f"Path {i} (L={path.length:.2f})", lw=2)

        ports = np.array([p['position'] for p in self.collector.ports])
        ax.scatter(ports[:,0], ports[:,1], ports[:,2], c='red', s=50, label="Collector Ports")
        ax.set_xlim(0, self.world.bounds[0]); ax.set_ylim(0, self.world.bounds[1]); ax.set_zlim(0, self.world.bounds[2])
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.legend(); plt.show()

# --- SETUP ---
starts = [
    [0, 0, 20],
    [0, 5, 20],
    [0, 10, 20],
    [0, 15, 20]
]
manual_pairs = [(0, 0), (1, 1), (2, 2), (3, 3)] # Manual Pairing
world = WorldOrder(bounds=(25, 25, 25))
coll = Collector(pos=(10, 20, 5), direction=(0, 1, 0), num_ports=4, radius=3)
manager = RouterManager(world, coll, starts, manual_pairs)

# --- MASTER ROUTE (Step 4) ---
print("Routing Master Path...")
master = manager.generate_path(0, stub_len=3.0, bias=0.4)
manager.ref_len = master.length
manager.paths[0] = master
print(f"Reference Length Set: {manager.ref_len:.2f}")

# --- FEEDBACK LOOP (Step 5) ---
# Naming corrected here to match 'stub_len' in method definition
current_params = {i: {"stub_len": 3.0, "bias": 0.4} for i in range(len(manual_pairs))}

while True:
    for i in range(1, len(manual_pairs)):
        # **current_params[i] now correctly passes 'stub_len'
        p = manager.generate_path(i, **current_params[i])
        manager.paths[i] = p
        diff = p.length - manager.ref_len
        print(f"Path {i}: Len={p.length:.2f} | Diff vs Ref={diff:+.2f}")

    manager.plot()

    print("\n[Feedback Adjustment]")
    cmd = input("Enter 'idx stub_len bias' to tune (e.g. '1 5.5 0.3') or 'q' to quit: ")
    if cmd.lower() == 'q': break
    try:
        idx, s, b = map(float, cmd.split())
        current_params[int(idx)] = {"stub_len": s, "bias": b}
    except ValueError:
        print("Invalid format. Use: index stub bias")