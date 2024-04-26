import os
import re
import pandas as pd

from utils import parse_results
from heatmap import generate_heatmap
from single_model_regression import generate_regression_plots
from violin import create_evaluation_violin
from scatter_eval_models import generate_scatter_eval_models_plot


def write_html_header(f):
    f.write('<head>\n')
    f.write(f'<title>Results for Run {id}</title>\n')
    f.write('<link rel="icon" href="favicon.png">\n')
    f.write(
        '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome'
        '.min.css">\n')
    f.write('<style>\n')
    f.write('body {font-family: Avenir, sans-serif;}\n')
    f.write('ul {list-style-type: none;}\n')
    f.write('li {padding: 5px;}\n')
    f.write(
        '.sidenav {\nwidth: 200px;\nposition: fixed;\nz-index: 1;\ntop: 20px;\nleft: 10px;\nbackground: '
        '#eee;\noverflow-x: hidden;\npadding: 8px 0;\n}\n')
    f.write(
        '.sidenav a {\npadding: 6px 8px 6px 16px;\ntext-decoration: none;\nfont-size: 20px;\ncolor: '
        '#2196F3;\ndisplay: block;\n}\n')
    f.write('.sidenav a:hover {\ncolor: #0645AD;\n}\n')
    f.write('.main {\nmargin-left: 220px;\nfont-size: 16px;\npadding: 0px 10 px;}\n')
    f.write(
        '@media screen and (max-height: 450px) {\n.sidenav {padding-top: 15px;}\n.sidenav a {font-size: 18px;}\n}\n')
    f.write(
        '.button {\nfont: 12px Avenir; text-decoration: none; background-color: DodgerBlue; color: white; '
        'padding: 2px 6px 2px 6px; \n}\n')
    f.write('</style>\n')
    f.write('</head>\n')


def write_sidebar(f):
    f.write('<div class="sidenav">\n')
    f.write(
        '<img src="favicon.png" width="80px" height="80px" alt="Logo" style="margin-left: auto; margin-right: '
        'auto; display: block; width=50%;">\n')
    f.write('<p style="text-align: center; color: #737272; font-size: 12px; ">v0.1</p>\n')
    f.write('<a href="#violin">Violin Plot</a>\n')
    f.write('<a href="#heatmap">Heatmap</a>\n')
    f.write('<a href="#regression_plots">Regression plots: True vs. Predicted</a>\n')
    f.write('<a href="#corr_comp">Correlation comparison</a>\n')
    f.write(
        '<a href="#corr_comp_drug" style="font-size: 14px; padding: 6px 8px 6px 26px">Correlation comparison per drug</a>\n')
    f.write(
        '<a href="#corr_comp_cls" style="font-size: 14px; padding: 6px 8px 6px 26px">Correlation comparison per cell line</a>\n')
    f.write('</div>\n')


def write_violins_and_heatmaps(f, setting, plot='Violin'):
    if plot == 'Violin':
        nav_id = 'violin'
        dir_name = 'violin_plots'
        prefix = 'violinplot'
    else:
        nav_id = 'heatmap'
        dir_name = 'heatmaps'
        prefix = 'heatmap'
    f.write(f'<h2 id="{nav_id}">{plot} Plots of Performance Measures over CV runs</h2>\n')
    f.write(f'<h3>{plot} plots comparing all models</h3>\n')
    f.write(
        f'<iframe src="{dir_name}/{prefix}_algorithms_{setting}.html" width="100%" height="80%" frameBorder="0"></iframe>\n')
    f.write(f'<h3>{plot} plots comparing all models with normalized metrics</h3>\n')
    f.write(
        f'<iframe src="{dir_name}/{prefix}_algorithms_{setting}_normalized.html" width="100%" height="80%" frameBorder="0"></iframe>\n')
    plot_list = [f for f in os.listdir(f'../results/{run_id}/{dir_name}') if setting in f
                 and f != f'{prefix}_algorithms_{setting}.html'
                 and f != f'{prefix}_algorithms_{setting}_normalized.html']
    f.write(f'<h3>{plot} plots comparing performance measures for tests within each model</h3>\n')
    f.write('<ul>')
    for plot in plot_list:
        f.write(f'<li><a href="{dir_name}/{plot}" target="_blank">{plot}</a></li>\n')
    f.write('</ul>\n')


def write_scatter_eval_models(f, setting, group_by):
    group_comparison_list = [f for f in os.listdir(f'../results/{run_id}/scatter_eval_models') if
                            setting in f and group_by in f]
    if len(group_comparison_list) > 0:
        f.write('<h3 id="corr_comp_drug">Drug-wise comparison</h3>\n')
        f.write('<h4>Overall comparison between models</h4>\n')
        f.write(f'<iframe src="scatter_eval_models/scatter_eval_models_{group_by}_overall_{setting}.html" width="100%" height="100%" frameBorder="0"></iframe>\n')
        f.write('<h4>Comparison between all models, dropdown menu</h4>\n')
        f.write(f'<iframe src="scatter_eval_models/scatter_eval_models_{group_by}_{setting}.html" width="100%" height="100%" frameBorder="0"></iframe>\n')
        f.write('<h4>Comparisons per model</h4>\n')
        f.write('<ul>\n')
        group_comparison_list = [elem for elem in group_comparison_list
                                 if elem != f'scatter_eval_models_{group_by}_{setting}.html' and
                                 elem != f'scatter_eval_models_{group_by}_overall_{setting}.html']
        group_comparison_list.sort()
        for group_comparison in group_comparison_list:
            f.write(f'<li><a href="scatter_eval_models/{group_comparison}" target="_blank">{group_comparison}</a></li>\n')
        f.write('</ul>\n')




def create_html(run_id, setting):
    # copy images to the results directory
    os.system(f'cp favicon.png ../results/{run_id}')
    os.system(f'cp nf-core-drugresponseeval_logo_light.png ../results/{run_id}')
    with open(f'../results/{run_id}/{setting}.html', 'w') as f:
        f.write('<html>\n')
        write_html_header(f)
        write_sidebar(f)

        f.write('<body>\n')
        f.write('<div class="main">\n')
        f.write('<img src="nf-core-drugresponseeval_logo_light.png" width="364px" height="100px" alt="Logo">\n')
        f.write(f'<h1>Results for {run_id}: {setting}</h1>\n')

        write_violins_and_heatmaps(f, setting, plot='Violin')
        write_violins_and_heatmaps(f, setting, plot='Heatmap')

        f.write('<h2 id="regression_plots">Regression plots</h2>\n')
        f.write('<ul>\n')
        file_list = [f for f in os.listdir(f'../results/{run_id}/regression_plots') if setting in f]
        file_list.sort()
        for file in file_list:
            f.write(f'<li><a href="regression_plots/{file}" target="_blank">{file}</a></li>\n')
        f.write('</ul>\n')

        f.write('<h2 id="corr_comp">Comparison of correlation metrics</h2>\n')
        write_scatter_eval_models(f, setting, 'drug')
        write_scatter_eval_models(f, setting, 'cell_line')
        f.write('</div>\n')
        f.write('</body>\n')
        f.write('</html>\n')


def create_index_html(run_id):
    with open(f'../results/{run_id}/index.html', 'w') as f:
        f.write('<html>\n')
        write_html_header(f)
        write_sidebar(f)
        f.write('<body>\n')
        f.write('<div class="main">\n')
        f.write('<img src="nf-core-drugresponseeval_logo_light.png" width="364px" height="100px" alt="Logo">\n')
        f.write(f'<h1>Results for {run_id}</h1>\n')
        f.write('<h2>Available settings</h2>\n')
        settings = [f for f in os.listdir(f'../results/{run_id}') if f.endswith('.html')]
        settings.sort()
        f.write('<ul>\n')
        for setting in settings:
            f.write(f'<li><a href="{setting}" target="_blank">{setting}</a></li>\n')
        f.write('</ul>\n')
        f.write('</div>\n')
        f.write('</body>\n')
        f.write('</html>\n')


def prep_results(run_id):
    # eval_results, eval_results_per_drug, eval_results_per_cell_line, t_vs_p = parse_results(
    #    run_id)
    eval_results = pd.read_csv(f'../results/{run_id}/evaluation_results.csv', index_col=0)
    eval_results_per_drug = pd.read_csv(f'../results/{run_id}/evaluation_results_per_drug.csv', index_col=0)
    eval_results_per_cell_line = pd.read_csv(f'../results/{run_id}/evaluation_results_per_cell_line.csv',
                                             index_col=0)
    t_vs_p = pd.read_csv(f'../results/{run_id}/true_vs_pred.csv', index_col=0)
    # add variables
    # split the index by "_" into: algorithm, randomization, setting, split, CV_split
    new_columns = eval_results.index.str.split('_', expand=True).to_frame()
    new_columns.columns = ['algorithm', 'rand_setting', 'LPO_LCO_LDO', 'split', 'CV_split']
    new_columns.index = eval_results.index
    eval_results = pd.concat([new_columns.drop('split', axis=1), eval_results], axis=1)
    eval_results_per_drug[['algorithm', 'rand_setting', 'LPO_LCO_LDO', 'split', 'CV_split']] = eval_results_per_drug[
        'model'].str.split(
        '_', expand=True)
    eval_results_per_cell_line[['algorithm', 'rand_setting', 'LPO_LCO_LDO', 'split', 'CV_split']] = \
    eval_results_per_cell_line['model'].str.split(
        '_', expand=True)
    t_vs_p[['algorithm', 'rand_setting', 'LPO_LCO_LDO', 'split', 'CV_split']] = t_vs_p['model'].str.split(
        '_', expand=True)

    return eval_results, eval_results_per_drug, eval_results_per_cell_line, t_vs_p


def draw_violin_and_heatmap(df, run_id, plotname, whole_name=False, normalized_metrics=False):
    if not os.path.exists(f'../results/{run_id}/violin_plots'):
        os.mkdir(f'../results/{run_id}/violin_plots')
    if not os.path.exists(f'../results/{run_id}/heatmaps'):
        os.mkdir(f'../results/{run_id}/heatmaps')
    fig = create_evaluation_violin(df, normalized_metrics=normalized_metrics, whole_name=whole_name)
    fig.write_html(f'../results/{run_id}/violin_plots/violinplot_{plotname}.html')
    fig = generate_heatmap(df, normalized_metrics=normalized_metrics, whole_name=whole_name)
    fig.write_html(f'../results/{run_id}/heatmaps/heatmap_{plotname}.html')


def draw_scatter_grids_per_group(eval_res_group, group_by, setting, run_id):
    if not os.path.exists(f'../results/{run_id}/scatter_eval_models'):
        os.mkdir(f'../results/{run_id}/scatter_eval_models')
    eval_res_group_subset = eval_res_group[eval_res_group['LPO_LCO_LDO'] == setting]
    eval_res_group_models = eval_res_group_subset[eval_res_group_subset['rand_setting'] == 'predictions']
    # draw plots for comparison between all models
    fig, fig_overall = generate_scatter_eval_models_plot(eval_res_group_models, metric='Pearson',
                                                         color_by=group_by)
    fig.write_html(f'../results/{run_id}/scatter_eval_models/scatter_eval_models_{group_by}_{setting}.html')
    fig_overall.write_html(
        f'../results/{run_id}/scatter_eval_models/scatter_eval_models_{group_by}_overall_{setting}.html')

    # draw plots per model: compare between original model and models with modification
    for algorithm in eval_res_group_models['algorithm'].unique():
        eval_res_group_algorithm = eval_res_group_subset[eval_res_group_subset['algorithm'] == algorithm]
        fig, fig_overall = generate_scatter_eval_models_plot(eval_res_group_algorithm, metric='Pearson',
                                                             color_by=group_by)
        fig.write_html(
            f'../results/{run_id}/scatter_eval_models/scatter_eval_models_{group_by}_{algorithm}_{setting}.html')
        fig_overall.write_html(
            f'../results/{run_id}/scatter_eval_models/scatter_eval_models_{group_by}_overall_{algorithm}_{setting}.html')


if __name__ == "__main__":
    # Load the dataset
    run_id = 'my_run_id'
    evaluation_results, evaluation_results_per_drug, evaluation_results_per_cell_line, true_vs_pred = prep_results(
        run_id)

    settings = evaluation_results['LPO_LCO_LDO'].unique()

    for setting in settings:
        print(f'Generating report for {setting} ...')
        eval_results_subset = evaluation_results[evaluation_results['LPO_LCO_LDO'] == setting]
        true_vs_pred_subset = true_vs_pred[true_vs_pred['LPO_LCO_LDO'] == setting]

        # only draw figures for 'real' predictions comparing all models
        eval_results_algorithms = eval_results_subset[eval_results_subset['rand_setting'] == 'predictions']
        draw_violin_and_heatmap(eval_results_algorithms, run_id, f'algorithms_{setting}')

        # draw the same figures but with drug/cell-line normalized metrics
        draw_violin_and_heatmap(eval_results_algorithms, run_id, f'algorithms_{setting}_normalized', normalized_metrics=True)

        # draw figures for each algorithm with all randomizations etc
        for algorithm in eval_results_algorithms['algorithm'].unique():
            eval_results_algorithm = eval_results_subset[eval_results_subset['algorithm'] == algorithm]
            draw_violin_and_heatmap(eval_results_algorithm, run_id, f'{algorithm}_{setting}', whole_name=True)

        if setting == 'LPO' or setting == 'LCO':
            draw_scatter_grids_per_group(eval_res_group=evaluation_results_per_drug, group_by='drug', setting=setting,
                                         run_id=run_id)
            generate_regression_plots(true_vs_pred_subset, run_id, group_by='cell_line')
            generate_regression_plots(true_vs_pred_subset, run_id, group_by='cell_line', normalize=True)

        if setting == 'LPO' or setting == 'LDO':
            draw_scatter_grids_per_group(eval_res_group=evaluation_results_per_cell_line, group_by='cell_line',
                                         setting=setting, run_id=run_id)
            generate_regression_plots(true_vs_pred_subset, run_id, group_by='drug')
            generate_regression_plots(true_vs_pred_subset, run_id, group_by='drug', normalize=True)
        create_html(run_id, setting)
    create_index_html(run_id)