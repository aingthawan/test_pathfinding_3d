import numpy as np
import matplotlib.pyplot as plt

class WorldOrder:
    def __init__(self, bounds=(25, 25, 25)):
        self.bounds = np.array(bounds)

    def is_within_bounds(self, point):
        """Step 4-4: Boundary check logic."""
        return np.all(point >= 0) and np.all(point <= self.bounds)

class Collector:
    def __init__(self, pos=(10, 20, 5), direction=(0, 1, 0), num_ports=4, radius=3.0):
        self.pos = np.array(pos)
        self.direction = self._normalize(np.array(direction))
        self.n = num_ports
        self.r = radius
        self.ports = self._generate_ports()

    def _normalize(self, v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def _get_orthogonal_basis(self):
        pick = np.array([1, 0, 0]) if abs(self.direction[0]) < 0.9 else np.array([0, 1, 0])
        u = self._normalize(np.cross(self.direction, pick))
        v = np.cross(self.direction, u)
        return u, v

    def _generate_ports(self):
        ports = []
        u, v = self._get_orthogonal_basis()
        angle_step = 2 * np.pi / self.n
        for i in range(self.n):
            theta = i * angle_step
            radial_offset = self.r * (np.cos(theta) * u + np.sin(theta) * v)
            ports.append({'id': i, 'position': self.pos + radial_offset, 'normal': self.direction})
        return ports

class SplinePath:
    def __init__(self, radius=1.0):
        self.coords = []
        self.length = 0.0
        self.radius = radius # Added radius for plotting

    def update_path(self, new_coords):
        self.coords = np.array(new_coords)
        diffs = np.diff(self.coords, axis=0)
        self.length = np.sum(np.sqrt((diffs**2).sum(axis=1)))

class RouterManager:
    def __init__(self, world, collector, starts, pairings):
        self.world = world
        self.collector = collector
        self.starts = np.array(starts)
        self.pairings = pairings
        self.paths = {}
        self.ref_len = 0.0

    def generate_path(self, pair_idx, stub_len, bias, radius=1.0):
        """Step 4-1 & 5-1: Stub-Bridge routing."""
        s_idx, p_idx = self.pairings[pair_idx]
        start_p = self.starts[s_idx]
        port = self.collector.ports[p_idx]

        s_dir = np.array([1, 0, 0])
        e_dir = port['normal']

        p_stub_out = start_p + (s_dir * stub_len)
        p_stub_in = port['position'] - (e_dir * stub_len)

        dist = np.linalg.norm(p_stub_in - p_stub_out)
        mult = dist * bias
        t = np.linspace(0, 1, 30)[:, None]
        cp0, cp3 = p_stub_out, p_stub_in
        cp1, cp2 = cp0 + (s_dir * mult), cp3 - (e_dir * mult)

        bridge = (1-t)**3*cp0 + 3*(1-t)**2*t*cp1 + 3*(1-t)*t**2*cp2 + t**3*cp3
        full_coords = np.vstack([start_p, p_stub_out, bridge, p_stub_in, port['position']])

        path = SplinePath(radius=radius)
        path.update_path(full_coords)

        for pt in path.coords:
            if not self.world.is_within_bounds(pt):
                print(f"--- WARNING: Path {pair_idx} out of bounds! ---")
                break
        return path

    def balance_lengths(self, tolerance=0.1):
        """
        Step 5-feedback: Automatically adjusts stub_len to match reference length.
        """
        print(f"\n--- Balancing Paths to Target: {self.ref_len:.2f} ---")

        for i in range(len(self.pairings)):
            # Skip the Master Path (index 0)
            if i == 0: continue

            # Start with a base configuration
            s_len = 2.0
            bias = 0.4

            # Iteratively increase stub_len to grow the pipe length
            attempts = 0
            while attempts < 100:
                p = self.generate_path(i, stub_len=s_len, bias=bias)

                # If we are close enough to the reference, stop
                if abs(p.length - self.ref_len) < tolerance:
                    break

                # If still too short, grow the stub
                if p.length < self.ref_len:
                    s_len += 0.1
                else:
                    # If it somehow got too long, nudge back and stop
                    s_len -= 0.05
                    break

                attempts += 1

            self.paths[i] = p
            diff = p.length - self.ref_len
            print(f"Path {i} Optimized: Stub={s_len:.2f} | Final Len={p.length:.2f} (Diff: {diff:+.2f})")

    def plot(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot BBox Cage for reference
        b = self.world.bounds
        for s, e in [([0,0,0],[b[0],0,0]), ([0,0,0],[0,b[1],0]), ([0,0,0],[0,0,b[2]])]:
            ax.plot([s[0],e[0]], [s[1],e[1]], [s[2],e[2]], color='gray', alpha=0.3, ls='--')

        # Plot Thick Paths
        for i, path in self.paths.items():
            pts = path.coords
            # Draw the center line
            ax.plot(pts[:,0], pts[:,1], pts[:,2], color='black', alpha=0.5, lw=1)

            # Draw "Thickness" using scatter spheres
            # S is proportional to radius^2 in points
            ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=(path.radius * 20)**2, alpha=0.2, label=f"P{i} (L={path.length:.1f})")

        ports = np.array([p['position'] for p in self.collector.ports])
        ax.scatter(ports[:,0], ports[:,1], ports[:,2], c='red', s=100, label="Ports")

        ax.set_xlim(0, b[0]); ax.set_ylim(0, b[1]); ax.set_zlim(0, b[2])
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
coll = Collector(pos=(5, 20, 5), direction=(0, 1, 0), num_ports=4, radius=2)
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

    manager.balance_lengths(tolerance=0.05)
    manager.plot()

    print("\n[Feedback Adjustment]")
    cmd = input("Enter 'idx stub_len bias' to tune (e.g. '1 5.5 0.3') or 'q' to quit: ")
    if cmd.lower() == 'q':
        for i, path in manager.paths.items():
            print(i, path)
        break
    try:
        idx, s, b = map(float, cmd.split())
        current_params[int(idx)] = {"stub_len": s, "bias": b}
    except ValueError:
        print("Invalid format. Use: index stub bias")