# app.py — Flask-сервер с ядром имитационного моделирования
from flask import Flask, render_template, request, jsonify
import random
import math
from collections import deque

app = Flask(__name__)

# ------------------ Ядро симуляции (Python) ------------------
class Job:
    __slots__ = ('id',)
    def __init__(self, job_id):
        self.id = job_id

class Node:
    def __init__(self, idx, mu, channels, label):
        self.idx = idx
        self.mu = mu
        self.channels = channels
        self.label = label
        self.queue = deque()
        self.busy = 0

class SimulationCore:
    def __init__(self, nodes_data, P, net_type, total_K=None, lambda0=None):
        self.nodes = [Node(i, d['mu'], d['channels'], d['label']) for i, d in enumerate(nodes_data)]
        self.P = P  # матрица переходов
        self.net_type = net_type  # 'open' или 'closed'
        self.total_K = total_K
        self.lambda0 = lambda0
        self.time = 0.0
        self.events = []  # список (time, type, node_idx)
        self.stats = {
            'arrivals': 0,
            'completions': 0,
            'queue_integral': [0.0]*len(self.nodes),
            'busy_integral': [0.0]*len(self.nodes),
            'last_update': 0.0
        }
        # Новое: счётчики переходов между узлами
        self.transitions = [[0]*len(self.nodes) for _ in range(len(self.nodes))]
        self.rng = random.Random(42)  # фиксированный seed для воспроизводимости
        self._init_jobs()

    def _init_jobs(self):
        if self.net_type == 'closed':
            K = self.total_K
            for k in range(K):
                job = Job(f"init_{k}")
                target = self.rng.randint(0, len(self.nodes)-1)
                node = self.nodes[target]
                if node.busy < node.channels:
                    node.busy += 1
                    self._schedule_departure(target)
                else:
                    node.queue.append(job)
        else:
            # открытая: первое прибытие
            self._schedule_arrival(0.0)

    def _schedule_arrival(self, delay=0.0):
        if self.net_type != 'open':
            return
        inter = -math.log(1 - self.rng.random()) / self.lambda0
        self.events.append((self.time + delay + inter, 'arrival', None))
        self.events.sort(key=lambda x: x[0])

    def _schedule_departure(self, node_idx):
        mu = self.nodes[node_idx].mu
        service = -math.log(1 - self.rng.random()) / mu
        self.events.append((self.time + service, 'departure', node_idx))
        self.events.sort(key=lambda x: x[0])

    def _select_next(self, from_idx):
        probs = self.P[from_idx]
        r = self.rng.random()
        cum = 0.0
        for j, p in enumerate(probs):
            cum += p
            if r < cum:
                return j
        return len(probs)-1

    def _update_integrals(self, dt):
        for i, node in enumerate(self.nodes):
            self.stats['queue_integral'][i] += len(node.queue) * dt
            self.stats['busy_integral'][i] += node.busy * dt

    def run_until(self, max_time=100.0, max_events=5000):
        event_count = 0
        while self.time < max_time and event_count < max_events:
            if not self.events:
                if self.net_type == 'open':
                    self._schedule_arrival()
                else:
                    # замкнутая: запускаем обслуживание из очередей если есть
                    for i, node in enumerate(self.nodes):
                        while node.busy < node.channels and node.queue:
                            node.queue.popleft()
                            node.busy += 1
                            self._schedule_departure(i)
                    if not self.events:
                        break
            ev_time, ev_type, node_idx = self.events.pop(0)
            dt = ev_time - self.time
            if dt > 0:
                self._update_integrals(dt)
            self.time = ev_time

            if ev_type == 'arrival':
                self.stats['arrivals'] += 1
                target = self.rng.randint(0, len(self.nodes)-1)
                node = self.nodes[target]
                job = Job(f"arr_{self.stats['arrivals']}")
                if node.busy < node.channels:
                    node.busy += 1
                    self._schedule_departure(target)
                else:
                    node.queue.append(job)
                self._schedule_arrival()

            elif ev_type == 'departure':
                node = self.nodes[node_idx]
                node.busy -= 1
                self.stats['completions'] += 1
                # маршрутизация
                if self.net_type == 'open':
                    exit_prob = 1.0 - sum(self.P[node_idx])
                    if self.rng.random() < exit_prob:
                        pass  # покидает сеть
                    else:
                        next_node = self._select_next(node_idx)
                        self.transitions[node_idx][next_node] += 1
                        nxt = self.nodes[next_node]
                        if nxt.busy < nxt.channels:
                            nxt.busy += 1
                            self._schedule_departure(next_node)
                        else:
                            nxt.queue.append(Job(f"cont_{self.time}"))
                else:
                    # замкнутая: всегда переходим
                    next_node = self._select_next(node_idx)
                    self.transitions[node_idx][next_node] += 1
                    nxt = self.nodes[next_node]
                    if nxt.busy < nxt.channels:
                        nxt.busy += 1
                        self._schedule_departure(next_node)
                    else:
                        nxt.queue.append(Job(f"cont_{self.time}"))

                # запускаем следующую из очереди этого узла
                if node.queue:
                    node.queue.popleft()
                    node.busy += 1
                    self._schedule_departure(node_idx)

            event_count += 1

        # расчёт статистики
        T = max(self.time, 0.001)
        result = {
            'time': self.time,
            'arrivals': self.stats['arrivals'],
            'completions': self.stats['completions'],
            'nodes': [],
            'flows': []  # интенсивности потоков между узлами
        }
        # Вычисляем интенсивности потоков
        flows = []
        for i in range(len(self.nodes)):
            row = []
            for j in range(len(self.nodes)):
                flow = self.transitions[i][j] / T
                row.append(flow)
            flows.append(row)
        result['flows'] = flows

        for i, node in enumerate(self.nodes):
            Lq = self.stats['queue_integral'][i] / T
            B = self.stats['busy_integral'][i] / T
            rho = B / node.channels
            L = Lq + B
            # пропускная способность узла (число завершений в узле) / T
            thru = self.stats['completions'] / T if self.stats['completions']>0 else 0.0
            W = L / (thru + 1e-9)
            result['nodes'].append({
                'idx': i,
                'L': L,
                'Lq': Lq,
                'rho': rho,
                'W': W,
                'queue_len': len(node.queue),
                'busy': node.busy
            })
        return result

# ------------------ Маршруты Flask ------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    data = request.json
    nodes_data = data['nodes']
    P = data['P']
    net_type = data['net_type']
    total_K = data.get('total_K', 5)
    lambda0 = data.get('lambda0', 0.8)
    sim_time = data.get('sim_time', 50.0)
    core = SimulationCore(nodes_data, P, net_type, total_K, lambda0)
    result = core.run_until(max_time=sim_time, max_events=10000)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)