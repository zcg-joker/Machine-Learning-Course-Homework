import argparse
import torch
import random
import logging
from config import Config
from model import NameRNN
from train import Trainer, create_char_maps, load_data, setup_logger, split_data
from predict import NameGenerator
from visualizer import Visualizer

def train_model(config):
    # 创建新的实验目录
    config.create_dirs()
    
    # 设置日志
    logger = setup_logger(config)
    logger.info(f"开始{config.model_type}模型训练...")
    
    # 记录训练参数
    logger.info("\n训练参数:")
    for param, value in config.get_training_params().items():
        logger.info(f"{param}: {value}")
    
    # 加载数据
    logger.info("\n加载数据...")
    names = load_data(config.data_path, logger)
    if len(names) == 0:
        logger.error("没有有效的训练数据！")
        return None, None, None
        
    char_to_idx, idx_to_char = create_char_maps()
    logger.info(f"字符表大小: {len(char_to_idx)}")
    
    # 打印数据示例
    logger.info("\n数据示例:")
    sample_size = min(5, len(names))
    for i, name in enumerate(random.sample(names, sample_size)):
        logger.info(f"示例 {i+1}: {name}")
        if config.model_type == 'reverse':
            logger.info(f"反向后: {''.join(reversed(name))}")
    
    # 创建模型
    logger.info("\n创建模型...")
    input_size = len(char_to_idx)
    output_size = len(char_to_idx)
    model = NameRNN(input_size, config.hidden_size, output_size, 
                    config.n_layers, config.bidirectional, config=config)
    
    # 划分数据集
    logger.info("\n划分数据集...")
    train_data, val_data, test_data = split_data(names)
    logger.info(f"训练集大小: {len(train_data)}")
    logger.info(f"验证集大小: {len(val_data)}")
    logger.info(f"测试集大小: {len(test_data)}")
    
    # 训练模型
    trainer = Trainer(model, config)
    history = trainer.train(train_data, val_data, char_to_idx, 
                          reverse=(config.model_type == 'reverse'))
    
    # 评估生成质量
    logger.info("\n评估生成质量...")
    scores, samples = trainer.evaluate_generation(test_data, char_to_idx)
    for metric, score in scores.items():
        logger.info(f"{metric}: {score:.4f}")
    
    # 可视化训练过程
    visualizer = Visualizer(config)
    logger.info("\n生成训练过程可视化...")
    visualizer.plot_training_curves(history)
    
    # 可视化数据分布
    logger.info("\n生成数据分布可视化...")
    visualizer.plot_character_distribution(names, 'first')
    visualizer.plot_character_distribution(names, 'last')
    visualizer.plot_name_length_distribution(names)
    
    # 保存当前实验目录
    config.save_exp_dir()
    logger.info(f"\n实验目录已保存: {config.result_dir}")
    
    return model, char_to_idx, idx_to_char

def generate_names(model, char_to_idx, idx_to_char, config):
    logger = logging.getLogger('name_generator')
    generator = NameGenerator(model, char_to_idx, idx_to_char, config)
    visualizer = Visualizer(config)
    
    # 记录生成参数
    logger.info("\n生成参数:")
    for param, value in config.get_generation_params().items():
        logger.info(f"{param}: {value}")
    
    while True:
        if config.model_type == 'forward':
            print("\n选择生成模式:")
            print("1. 从开头生成")
            print("2. 比较不同解码策略")
            print("3. 退出")
            
            choice = input("请输入选项 (1-3): ").strip()
            
            if choice == '3':
                break
                
            if choice == '2':
                start_seq = input("请输入起始字符（直接回车随机生成）: ").strip()
                if not start_seq:
                    start_seq = config.start_token
                
                logger.info(f"使用不同策略生成名字（起始序列: {start_seq}）...")
                results = {
                    'Greedy': generator.generate_greedy(start_seq),
                    'Temperature': generator.generate_temperature(start_seq, config.temperature),
                    'Beam Search': generator.generate_beam_search(start_seq, config.beam_size),
                    'Top-K': generator.generate_top_k(start_seq, config.top_k),
                    'Top-P': generator.generate_top_p(start_seq, config.top_p)
                }
                
                for method, (name, _, _) in results.items():
                    logger.info(f"{method}: {name}")
                
                visualizer.plot_generation_comparison(results)
                continue
            
            # 从开头生成
            seq = input("请输入起始字符（直接回车随机生成）: ").strip()
            if not seq:
                seq = config.start_token
            logger.info(f"从开头生成名字（起始序列: {seq}）...")
            name, probs, chars = generator.generate_temperature(seq)
            logger.info(f"生成的名字: {name}")
            
        else:  # reverse model
            print("\n选择生成模式:")
            print("1. 从结尾生成")
            print("2. 比较不同解码策略")
            print("3. 退出")
            
            choice = input("请输入选项 (1-3): ").strip()
            
            if choice == '3':
                break
            
            if choice == '2':
                end_seq = input("请输入结尾字符: ").strip()
                logger.info(f"使用不同策略生成名字（结尾序列: {end_seq}）...")
                
                # 对于反向模型，我们需要先反转输入序列
                end_seq_reversed = ''.join(reversed(end_seq))
                results = {
                    'Greedy': generator.generate_greedy(end_seq_reversed),
                    'Temperature': generator.generate_temperature(end_seq_reversed, config.temperature),
                    'Beam Search': generator.generate_beam_search(end_seq_reversed, config.beam_size),
                    'Top-K': generator.generate_top_k(end_seq_reversed, config.top_k),
                    'Top-P': generator.generate_top_p(end_seq_reversed, config.top_p)
                }
                
                # 反转生成的结果
                results = {
                    method: (''.join(reversed(name)), probs, chars)
                    for method, (name, probs, chars) in results.items()
                }
                
                for method, (name, _, _) in results.items():
                    logger.info(f"{method}: {name}")
                    logger.info(f"{method}（生成过程）: {''.join(reversed(name))}")
                
                visualizer.plot_generation_comparison(results)
                continue
            
            # 从结尾生成
            seq = input("请输入结尾字符: ").strip()
            logger.info(f"从结尾生成名字（结尾序列: {seq}）...")
            # 使用generator的temperature生成方法
            end_seq_reversed = ''.join(reversed(seq))
            name_reversed, probs, chars = generator.generate_temperature(end_seq_reversed)
            name = ''.join(reversed(name_reversed))
            logger.info(f"生成的名字: {name}")
            logger.info(f"生成过程（反向）: {name_reversed}")
        
        visualizer.plot_generation_process(name, probs, chars)

def main():
    parser = argparse.ArgumentParser(description='Name Generator')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--generate', action='store_true', help='Generate names')
    parser.add_argument('--model-type', type=str, choices=['forward', 'reverse'],
                       default='forward', help='Model type to train or use')
    args = parser.parse_args()
    
    config = Config()
    config.model_type = args.model_type
    
    if args.train:
        model, char_to_idx, idx_to_char = train_model(config)
    else:
        # 加载最新实验目录
        if not config.load_latest_exp_dir():
            print("错误：找不到已训练的实验！请先训练模型。")
            return
            
        # 设置日志
        logger = setup_logger(config)
        logger.info(f"加载已训练的{config.model_type}模型...")
        
        # 创建模型
        char_to_idx, idx_to_char = create_char_maps()
        input_size = len(char_to_idx)
        output_size = len(char_to_idx)
        model = NameRNN(input_size, config.hidden_size, output_size, 
                       config.n_layers, config.bidirectional, config=config)
        
        # 加载模型
        trainer = Trainer(model, config)
        if not trainer.load_model(reverse=(config.model_type == 'reverse')):
            logger.error(f"错误：找不到已训练的{config.model_type}模型！请先训练模型。")
            return
    
    if args.generate:
        generate_names(model, char_to_idx, idx_to_char, config)

if __name__ == '__main__':
    main() 