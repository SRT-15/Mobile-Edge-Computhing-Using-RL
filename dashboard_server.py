import os
import csv
import threading
from flask import Flask, send_from_directory, abort
from flask_socketio import SocketIO
from config import ALGOS, DATASETS, DEFAULT_ALGO, DEFAULT_DATASET
import train as trainer
HERE = os.path.dirname(os.path.abspath(__file__))
DASHBOARD = os.path.join(HERE, 'dashboard.html')
app = Flask(__name__)
app.config['SECRET_KEY'] = 'mec-combined-2026'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')
_state = {'running': False, 'stop': threading.Event(), 'config': {'algo': DEFAULT_ALGO, 'dataset': DEFAULT_DATASET, 'episodes': 200, 'seed': 0}}

@app.route('/')
def index():
    if not os.path.exists(DASHBOARD):
        return ('dashboard.html missing', 500)
    with open(DASHBOARD) as f:
        return f.read()

@app.route('/results/<path:fname>')
def results(fname):
    rdir = trainer.RESULTS_DIR
    if not os.path.exists(os.path.join(rdir, fname)):
        abort(404)
    return send_from_directory(rdir, fname)

def _worker(cfg):

    def on_episode(ep, reward):
        socketio.emit('episode', {'episode': ep, 'reward': round(reward, 4), 'algo': cfg['algo'], 'dataset': cfg['dataset'], 'comparable': cfg['dataset'] == 'synthetic'})

    def should_stop():
        return _state['stop'].is_set()
    try:
        (episodes, rewards, meta) = trainer.run_training(algo=cfg['algo'], dataset=cfg['dataset'], episodes=int(cfg['episodes']), seed=int(cfg['seed']), on_episode=on_episode, should_stop=should_stop, log=lambda *_: None)
        (csv_path, png_path) = trainer.save_results(episodes, rewards, meta)
        socketio.emit('done', {'meta': meta, 'plot_url': '/results/' + os.path.basename(png_path), 'csv': os.path.basename(csv_path), 'stopped': _state['stop'].is_set()})
    except Exception as e:
        socketio.emit('error', {'message': repr(e)})
    finally:
        _state['running'] = False
        _state['stop'].clear()

def _benchmark_worker(cfg):
    try:
        algos = cfg.get('bench_algos', list(ALGOS))
        seeds = cfg.get('seeds', [0, 1, 2])
        episodes = int(cfg.get('episodes', 200))
        dataset = cfg.get('dataset', DEFAULT_DATASET)
        (runs_csv, summary_csv) = trainer.run_benchmark(bench_algos=algos, dataset=dataset, episodes=episodes, seeds=seeds, n_agents=None, out_dir=trainer.RESULTS_DIR, last_k=5)
        summary_rows = []
        with open(summary_csv, newline='') as f:
            for row in csv.DictReader(f):
                summary_rows.append(row)
        socketio.emit('benchmark_done', {'dataset': dataset, 'episodes': episodes, 'seeds': seeds, 'algos': algos, 'runs_csv': os.path.basename(runs_csv), 'summary_csv': os.path.basename(summary_csv), 'summary_rows': summary_rows})
    except Exception as e:
        socketio.emit('error', {'message': repr(e)})
    finally:
        _state['running'] = False
        _state['stop'].clear()

@socketio.on('connect')
def on_connect():
    socketio.emit('status', {'message': 'connected', 'algos': list(ALGOS), 'datasets': list(DATASETS)})
    socketio.emit('config_changed', _state['config'])

@socketio.on('request_config')
def on_request_config(_data=None):
    socketio.emit('config_changed', _state['config'])

@socketio.on('select_config')
def on_select_config(data):
    for k in ('algo', 'dataset', 'episodes', 'seed'):
        if isinstance(data, dict) and k in data:
            _state['config'][k] = data[k]
    socketio.emit('config_changed', _state['config'])

@socketio.on('start_training')
def on_start(data=None):
    if _state['running']:
        socketio.emit('status', {'message': 'already running'})
        return
    data = data or {}
    cfg = dict(_state['config'])
    for k in ('algo', 'dataset', 'episodes', 'seed'):
        if k in data:
            cfg[k] = data[k]
    if cfg['algo'] not in ALGOS or cfg['dataset'] not in DATASETS:
        socketio.emit('error', {'message': f'bad config {cfg}'})
        return
    _state['config'] = cfg
    _state['running'] = True
    _state['stop'].clear()
    socketio.emit('status', {'message': f"starting {cfg['algo']}/{cfg['dataset']}", 'comparable': cfg['dataset'] == 'synthetic'})
    socketio.start_background_task(_worker, cfg)

@socketio.on('start_benchmark')
def on_start_benchmark(data=None):
    if _state['running']:
        socketio.emit('status', {'message': 'already running'})
        return
    data = data or {}
    dataset = data.get('dataset', _state['config'].get('dataset', DEFAULT_DATASET))
    episodes = int(data.get('episodes', _state['config'].get('episodes', 200)))
    algos = data.get('bench_algos', list(ALGOS))
    seeds = data.get('seeds', [0, 1, 2])
    if isinstance(algos, str):
        algos = [x.strip() for x in algos.split(',') if x.strip()]
    if isinstance(seeds, str):
        seeds = [int(x.strip()) for x in seeds.split(',') if x.strip()]
    if not isinstance(algos, list) or not all((a in ALGOS for a in algos)):
        socketio.emit('error', {'message': f'bad benchmark algos {algos}'})
        return
    if not isinstance(seeds, list) or not all((isinstance(s, int) for s in seeds)):
        socketio.emit('error', {'message': f'bad benchmark seeds {seeds}'})
        return
    if dataset not in DATASETS:
        socketio.emit('error', {'message': f'bad benchmark dataset {dataset}'})
        return
    cfg = {'dataset': dataset, 'episodes': episodes, 'bench_algos': algos, 'seeds': seeds}
    _state['running'] = True
    _state['stop'].clear()
    socketio.emit('status', {'message': f'starting benchmark ({len(algos)} algos x {len(seeds)} seeds)', 'comparable': dataset == 'synthetic'})
    socketio.start_background_task(_benchmark_worker, cfg)

@socketio.on('stop_training')
def on_stop(_data=None):
    if _state['running']:
        _state['stop'].set()
        socketio.emit('status', {'message': 'stop requested'})
    else:
        socketio.emit('status', {'message': 'nothing running'})
if __name__ == '__main__':
    print('\n' + '=' * 52)
    print('  MEC-DRL Combined Dashboard')
    print('  Open:  http://localhost:5001')
    print('  Default: MAA2C / synthetic (known-good path)')
    print('=' * 52 + '\n')
    socketio.run(app, host='0.0.0.0', port=5001, debug=False, allow_unsafe_werkzeug=True)
