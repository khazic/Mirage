import click
from .evaluate_task import run_evaluation
from .test_api import test_single_image
from .utils.config import load_config
import logging

@click.group()
def cli():
    """Mmoment CLI tool for evaluating multi-modal models"""
    pass

@cli.command()
@click.option('--config', '-c', default='config.json', help='Path to config file')
@click.option('--task', '-t', default='negative', help='Task to evaluate')
def evaluate(config, task):
    """Run evaluation pipeline"""
    config_dict = load_config(config)
    logger = logging.getLogger(__name__)
    results = run_evaluation(config_dict, logger)
    click.echo(f"Evaluation completed. Results saved in {config_dict['data']['results_dir']}")

@cli.command()
@click.option('--config', '-c', default='config.json', help='Path to config file')
@click.option('--image', '-i', help='Path to test image')
def test_api(config, image):
    """Test API with single image"""
    config_dict = load_config(config)
    if image:
        config_dict["test_image_path"] = image
    test_single_image(config_dict)

if __name__ == '__main__':
    cli() 