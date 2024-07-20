import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    # 특정 값들을 적절한 타입으로 변환
    config['learning_rate'] = float(config['learning_rate'])
    config['batch_size'] = int(config['batch_size'])
    config['max_epochs'] = int(config['max_epochs'])
    config['checkpoint_save_top_k'] = int(config['checkpoint_save_top_k'])
    config['checkpoint_every_n_epochs'] = int(config['checkpoint_every_n_epochs'])
    config['logger_freq'] = int(config['logger_freq'])
    config['num_workers'] = int(config['num_workers'])
    config['devices'] = int(config['devices'])
    config['precision'] = int(config['precision'])
    return config