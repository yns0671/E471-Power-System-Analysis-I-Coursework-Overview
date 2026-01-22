import json
import numpy as np
import cmath
import os
import sys
import matplotlib.pyplot as plt

class PowerSystem:
    def __init__(self, topology_path, load_flow_path):
        # 1. Load Data
        print(f"Loading system from: {os.path.basename(topology_path)}")
        try:
            with open(topology_path, 'r') as f:
                self.topo = json.load(f)
            with open(load_flow_path, 'r') as f:
                self.case = json.load(f)
        except FileNotFoundError:
            print(f"\n❌ CRITICAL ERROR: File not found: {topology_path}")
            sys.exit()

        # 2. Extract System Base
        self.base_mva = self.topo['system_data']['s_base_mva']
        self.network_name = self.topo['system_data']['network_name']
        
        # 3. Dynamic Bus Mapping
        self.bus_list = self.topo['bus_data']
        self.n_bus = len(self.bus_list)
        
        self.id2idx = {} 
        self.idx2id = {}
        
        for i, bus in enumerate(self.bus_list):
            bid = bus['bus_id']
            self.id2idx[bid] = i
            self.idx2id[i] = bid

        # 4. Helper Maps (Gen ID -> Bus ID)
        self.gen_id_to_bus_idx = {}
        for gen in self.topo['gen_data']:
            gid = gen['gen_id']
            bid = gen['bus_id']
            if bid in self.id2idx:
                self.gen_id_to_bus_idx[gid] = self.id2idx[bid]
            
    def build_ybus(self):
        Y = np.zeros((self.n_bus, self.n_bus), dtype=complex)
        
        for line in self.topo['line_data']:
            f = self.id2idx[line['from_bus_id']]
            t = self.id2idx[line['to_bus_id']]
            v_base = self.bus_list[f]['voltage_level_kv']
            z_base = (v_base ** 2) / self.base_mva
            
            z_pu = complex(line['r_ohm'], line['x_ohm']) / z_base
            y_series = 1.0 / z_pu
            b_shunt = (line['b_total_mho'] * z_base) / 2.0
            
            Y[f, t] -= y_series
            Y[t, f] -= y_series
            Y[f, f] += y_series + complex(0, b_shunt)
            Y[t, t] += y_series + complex(0, b_shunt)

        for xfmr in self.topo['transformer_data']:
            f = self.id2idx[xfmr['from_bus_id']]
            t = self.id2idx[xfmr['to_bus_id']]
            z_pu = complex(xfmr['r_pu'], xfmr['x_pu'])
            y_series = 1.0 / z_pu
            a = xfmr['tap']
            phi = np.deg2rad(xfmr['phase_shift'])
            a_complex = a * cmath.exp(1j * phi)
            
            Y[f, f] += y_series / (abs(a_complex)**2)
            Y[t, t] += y_series
            Y[f, t] -= y_series / np.conj(a_complex)
            Y[t, f] -= y_series / a_complex

        if 'shunt_data' in self.topo:
            for shunt in self.topo['shunt_data']:
                idx = self.id2idx[shunt['bus_id']]
                g = shunt['p_mw'] / self.base_mva
                b = shunt['q_mvar'] / self.base_mva
                Y[idx, idx] += complex(g, b) 
        return Y

    def build_decoupled_matrices(self):
        Bp = np.zeros((self.n_bus, self.n_bus))
        Bpp = np.zeros((self.n_bus, self.n_bus))
        
        for line in self.topo['line_data']:
            f = self.id2idx[line['from_bus_id']]
            t = self.id2idx[line['to_bus_id']]
            v_base = self.bus_list[f]['voltage_level_kv']
            z_base = (v_base ** 2) / self.base_mva
            
            x_pu = line['x_ohm'] / z_base
            b_val = -1.0 / x_pu
            b_line_shunt = (line['b_total_mho'] * z_base) / 2.0
            
            Bp[f, t] -= b_val
            Bp[t, f] -= b_val
            Bp[f, f] += b_val
            Bp[t, t] += b_val
            
            Bpp[f, t] -= b_val
            Bpp[t, f] -= b_val
            Bpp[f, f] += b_val + b_line_shunt
            Bpp[t, t] += b_val + b_line_shunt

        for xfmr in self.topo['transformer_data']:
            f = self.id2idx[xfmr['from_bus_id']]
            t = self.id2idx[xfmr['to_bus_id']]
            b_val = -1.0 / xfmr['x_pu']
            a = xfmr['tap']
            
            Bp[f, t] -= b_val
            Bp[t, f] -= b_val
            Bp[f, f] += b_val
            Bp[t, t] += b_val
            
            Bpp[f, f] += b_val / (a**2)
            Bpp[t, t] += b_val
            Bpp[f, t] -= b_val / a
            Bpp[t, f] -= b_val / a

        if 'shunt_data' in self.topo:
            for shunt in self.topo['shunt_data']:
                idx = self.id2idx[shunt['bus_id']]
                b = shunt['q_mvar'] / self.base_mva
                Bpp[idx, idx] += b
                
        return Bp, Bpp

    def solve_fdlf(self, max_iter=100, tol=1e-4):
        print(f"--- Solving FDLF (Max Iter: {max_iter}) ---")
        
        Y = self.build_ybus()
        Bp, Bpp = self.build_decoupled_matrices()
        
        # Parse Scheduled Power
        P_sched = np.zeros(self.n_bus)
        Q_sched = np.zeros(self.n_bus)
        
        if 'load_data' in self.case:
            for load in self.case['load_data']:
                lid = load['load_id']
                bus_id = -1
                for t_load in self.topo['load_data']:
                    if t_load['load_id'] == lid:
                        bus_id = t_load['bus_id']
                        break
                if bus_id != -1 and bus_id in self.id2idx:
                    idx = self.id2idx[bus_id]
                    P_sched[idx] -= load['p_mw'] / self.base_mva
                    Q_sched[idx] -= load['q_mvar'] / self.base_mva

        if 'gen_data' in self.case:
            for gen in self.case['gen_data']:
                gid = gen['gen_id']
                if gid in self.gen_id_to_bus_idx:
                    idx = self.gen_id_to_bus_idx[gid]
                    P_sched[idx] += gen['p_mw'] / self.base_mva
        
        Vm = np.ones(self.n_bus)
        Va = np.zeros(self.n_bus)
        type_map = self.case['load_flow_type']
        slack_idx = -1
        pv_indices = []
        pq_indices = []
        
        for bid_str, type_str in type_map.items():
            idx = self.id2idx[int(bid_str)]
            if type_str == "SLACK":
                slack_idx = idx
                for s in self.case['slack_bus']:
                    if s['bus_id'] == int(bid_str):
                        Vm[idx] = s['vm_pu']
                        Va[idx] = np.deg2rad(s['va_degree'])
            elif type_str == "PV":
                pv_indices.append(idx)
                for gen in self.case['gen_data']:
                    if self.gen_id_to_bus_idx.get(gen['gen_id']) == idx:
                        Vm[idx] = gen['vm_pu']
            elif type_str == "PQ":
                pq_indices.append(idx)

        non_slack = [i for i in range(self.n_bus) if i != slack_idx]
        Bp_red = Bp[np.ix_(non_slack, non_slack)]
        Bpp_red = Bpp[np.ix_(pq_indices, pq_indices)]

        history_p = []
        history_q = []

        for it in range(max_iter):
            V_complex = Vm * np.exp(1j * Va)
            I_inj = Y @ V_complex
            S_calc = V_complex * np.conj(I_inj)
            P_calc = S_calc.real
            Q_calc = S_calc.imag
            
            dP = P_sched - P_calc
            dQ = Q_sched - Q_calc
            
            max_dP = np.max(np.abs(dP[non_slack]))
            max_dQ = np.max(np.abs(dQ[pq_indices])) if len(pq_indices) > 0 else 0.0
            
            history_p.append(max_dP)
            history_q.append(max_dQ)
            
            if it == 0 or it % 5 == 0:
                print(f"  Iter {it+1}: dP={max_dP:.4f}, dQ={max_dQ:.4f}")
            
            if max_dP < tol and max_dQ < tol:
                print(f"  ✅ Converged in {it+1} iterations!")
                break
                
            rhs_p = dP[non_slack] / Vm[non_slack]
            dTheta = -np.linalg.solve(Bp_red, rhs_p)
            Va[non_slack] += dTheta
            
            if len(pq_indices) > 0:
                rhs_q = dQ[pq_indices] / Vm[pq_indices]
                dV = -np.linalg.solve(Bpp_red, rhs_q)
                Vm[pq_indices] += dV
                
        return Vm, Va, history_p, history_q

    def calculate_losses(self, Vm, Va):
        """Calculates total system Active (MW) and Reactive (MVAR) losses."""
        Y = self.build_ybus()
        V_complex = Vm * np.exp(1j * Va)
        
        # Current Injection I = Y * V
        I_inj = Y @ V_complex
        
        # Power Injection S = V * conj(I)
        S_inj = V_complex * np.conj(I_inj)
        
        # Total Loss is sum of all net injections (Gen - Load)
        total_loss = np.sum(S_inj)
        
        P_loss_mw = total_loss.real * self.base_mva
        Q_loss_mvar = total_loss.imag * self.base_mva
        
        return P_loss_mw, Q_loss_mvar

    def save_plots(self, plots_dir, case_name, Vm, hist_p, hist_q):
        """Generates standard plots."""
        
        # 1. Sparsity Plot
        Y = self.build_ybus()
        plt.figure(figsize=(6, 6))
        plt.spy(Y, markersize=2)
        plt.title(f"Sparsity: {case_name} ({self.n_bus} Bus)")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{case_name}_sparsity.png"), dpi=150)
        plt.close()

        # 2. Convergence Plot (Basic)
        plt.figure(figsize=(6, 4))
        plt.semilogy(hist_p, label='Max dP', marker='')
        plt.semilogy(hist_q, label='Max dQ', marker='')
        plt.title(f"Convergence: {case_name}")
        plt.xlabel("Iteration")
        plt.ylabel("Mismatch (p.u.)")
        plt.legend()
        plt.grid(True, which="both", linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{case_name}_convergence.png"), dpi=150)
        plt.close()

        # 3. Voltage Profile
        plt.figure(figsize=(8, 4))
        bus_ids = [self.idx2id[i] for i in range(self.n_bus)]
        sorted_pairs = sorted(zip(bus_ids, Vm))
        s_ids, s_vm = zip(*sorted_pairs)
        
        if self.n_bus < 50:
            plt.xticks(range(len(s_ids)), s_ids, rotation=90, fontsize=8)
            plt.plot(range(len(s_ids)), s_vm, marker='o', markersize=3, color='purple', linestyle='-')
        else:
            plt.xticks([]) 
            plt.xlabel("Bus Index (Sorted)")
            plt.plot(range(len(s_ids)), s_vm, color='purple', linestyle='-', linewidth=1)
            
        plt.ylabel("Voltage (p.u.)")
        plt.title(f"Voltage Profile: {case_name}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{case_name}_voltage.png"), dpi=150)
        plt.close()
        
    def plot_convergence_analysis(self, plots_dir, case_name, hist_p, hist_q):
        """Generates detailed threshold analysis plot for report."""
        plt.figure(figsize=(8, 5))
        plt.semilogy(hist_p, label='P Mismatch', color='blue', linewidth=2)
        plt.semilogy(hist_q, label='Q Mismatch', color='red', linestyle='--', linewidth=1.5)
        
        thresholds = [1e-3, 1e-4, 1e-5, 1e-6]
        colors = ['gray', 'orange', 'green', 'purple']
        
        for tol, col in zip(thresholds, colors):
            plt.axhline(y=tol, color=col, linestyle=':', alpha=0.7)
            # Find crossover
            for i, val in enumerate(hist_p):
                if val < tol:
                    plt.plot(i, val, marker='o', color=col, markersize=8)
                    plt.text(i, tol*1.5, f"{i+1} iters", color=col, fontsize=9, fontweight='bold')
                    break

        plt.title(f"Convergence Analysis: {case_name}")
        plt.xlabel("Iteration")
        plt.ylabel("Max Mismatch (p.u.) [Log Scale]")
        plt.legend(loc='upper right', fontsize='small')
        plt.grid(True, which="both", linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{case_name}_convergence_analysis.png"), dpi=150)
        plt.close()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Setup Folders - CHANGE THIS PATH if needed
    base_path = r"C:\Users\Yunus Tosun\Desktop\EE471\project"
    plots_dir = os.path.join(base_path, "plots")
    
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"📁 Created plots folder at: {plots_dir}")

    cases = ["ieee14", "ieee30", "ieee57", "ieee118", "ieee300"]

    print(f"\n🚀 Starting Batch Processing for: {cases}")

    for case_name in cases:
        print(f"\n================ {case_name.upper()} ================")
        
        topo_file = os.path.join(base_path, f"{case_name}_topology.json")
        load_file = os.path.join(base_path, f"{case_name}_load_flow.json")
        
        if not os.path.exists(topo_file):
            print(f"⚠️ Skipping {case_name}: File not found ({topo_file})")
            continue

        try:
            sys_obj = PowerSystem(topo_file, load_file)
            
            # Use stricter tolerance for 118/300 to generate nice analysis plots
            run_tol = 1e-6 if case_name in ["ieee118", "ieee300"] else 1e-4
            
            Vm, Va, hist_p, hist_q = sys_obj.solve_fdlf(max_iter=100, tol=run_tol)
            
            # --- CALCULATE LOSSES ---
            P_loss, Q_loss = sys_obj.calculate_losses(Vm, Va)
            print(f"  ⚡ TOTAL SYSTEM LOSSES: P = {P_loss:.4f} MW,  Q = {Q_loss:.4f} MVAR")
            # ------------------------

            sys_obj.save_plots(plots_dir, case_name, Vm, hist_p, hist_q)
            
            # Generate Report-Specific Analysis Plots for large cases
            if case_name in ["ieee118", "ieee300"]:
                sys_obj.plot_convergence_analysis(plots_dir, case_name, hist_p, hist_q)
                print(f"  📊 Generated Convergence Analysis Plot")

        except Exception as e:
            print(f"❌ Error processing {case_name}: {e}")

    print("\n✅ Batch Processing Complete.")