import socket
import random

HOST = "127.0.0.1"
PORT = 5555

def run_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"Listening for Isaac on {HOST}:{PORT}...")
        
        conn, addr = s.accept()
        with conn:
            print(f"Connected to Isaac from {addr}!")
            # Use makefile() for clean line-by-line reading
            stream = conn.makefile("r", encoding="utf-8")
            
            step = 0
            while True:
                line = stream.readline()
                if not line:
                    print("Isaac disconnected.")
                    break
                
                # Parse incoming state packet
                parts = line.strip().split("|")
                tick, hp, max_hp, px, py, vx, vy, is_clear, enemy_cnt = parts[:9]
                
                if step % 15 == 0:  # Print once per simulated second
                    print(f"[Tick {tick}] HP: {hp}/{max_hp} | Pos: ({px}, {py}) | Enemies: {enemy_cnt}")
                
                # Test reset if Isaac dies (hp == 0)
                if int(hp) <= 0:
                    print("Isaac died! Sending RESET command...")
                    conn.sendall(b"RESET\n")
                    step = 0
                    continue
                
                # Random movement + shooting test
                mx = random.choice([-1.0, 0.0, 1.0])
                my = random.choice([-1.0, 0.0, 1.0])
                sx, sy = random.choice([(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)])
                
                action_str = f"{mx},{my},{sx},{sy}\n"
                conn.sendall(action_str.encode("utf-8"))
                step += 1

if __name__ == "__main__":
    run_server()