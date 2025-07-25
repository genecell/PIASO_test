import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

### plotting the ligand receptor interaction patterns
def plotLigandReceptorInteraction(
    interactions_df: pd.DataFrame,
    specificity_df: pd.DataFrame,
    cell_type_pairs: list,
    top_n: int = 50,
    y_max: int = 10,
    cell_type_sep: str = '@',
    ligand_receptor_sep: str = '-->',
    heatmap_height_ratio: float = 1.5,
    heatmap_cmap: str = 'Purples',
    shared_legend: bool = False,
    heatmap_vmax: float = None,
    save_path: str = None,
    fig_width: int = 24,
    fig_height_per_pair: int = 9,
    col_interaction_score: str = 'interaction_score',
    col_ligand_receptor_pair: str = 'ligandXreceptor',
    col_cell_type_pair: str = 'CellTypeXCellType',
    col_annotation: str = 'annotation',
    col_ligand: str = 'ligand',
    col_receptor: str = 'receptor'
):
    """
    Generates plots with a bar plot of top interactions and a 2-row heatmap 
    showing ligand (sender) and receptor (receiver) specificity.

    Args:
        interactions_df (pd.DataFrame): DataFrame with interaction scores.
        specificity_df (pd.DataFrame): DataFrame with gene specificity scores.
        cell_type_pairs (list): A list of 'CellTypeXCellType' strings to plot.
        top_n (int): The number of top interactions to display.
        y_max (int): The maximum y-axis value for the bar plot.
        cell_type_sep (str): The separator for sender/receiver cell types.
        ligand_receptor_sep (str): The separator for ligand/receptor genes.
        heatmap_height_ratio (float): The height ratio of the heatmap relative to the bar plot.
        heatmap_cmap (str): The colormap for the specificity heatmap.
        shared_legend (bool): If True, a single legend/colorbar is shown for all plots.
                              If False (default), each plot has its own.
        heatmap_vmax (float): The maximum value for the heatmap color scale. 
                              Defaults to the data's max if None.
        save_path (str, optional): Path to save the figure (e.g., 'plot.pdf'). 
                                   If None (default), the plot is not saved.
        fig_width (int): The total width of the figure.
        fig_height_per_pair (int): The height allocated for each cell type pair subplot.
        col_interaction_score (str): Column name for interaction scores.
        col_ligand_receptor_pair (str): Column name for ligand-receptor pair strings.
        col_cell_type_pair (str): Column name for cell type pair strings.
        col_annotation (str): Column name for pathway/annotation data.
        col_ligand (str): Column name for ligand after splitting.
        col_receptor (str): Column name for receptor after splitting.
    """
    n_pairs = len(cell_type_pairs)
    
    # Use a consistent 2-column layout. Left for plots, right for legends/cbars.
    fig = plt.figure(figsize=(fig_width, fig_height_per_pair * n_pairs))
    main_gs = gridspec.GridSpec(n_pairs, 2, figure=fig, hspace=0.8, width_ratios=[15, 3])

    # Create a consistent color palette for annotations
    all_annotations = sorted(list(interactions_df[col_annotation].unique()))
    color_palette = dict(zip(all_annotations, sns.color_palette('Paired', len(all_annotations))))

    # --- Pre-calculate global vmax for shared colorbar ---
    global_vmax = float('-inf')
    if shared_legend and heatmap_vmax is None:
        for cell_type_pair in cell_type_pairs:
            sender_cell_type, receiver_cell_type = cell_type_pair.split(cell_type_sep)
            top_interactions = interactions_df[interactions_df[col_cell_type_pair] == cell_type_pair].nlargest(top_n, col_interaction_score)
            if top_interactions.empty: continue

            filtered_interactions = top_interactions.copy()
            filtered_interactions[[col_ligand, col_receptor]] = filtered_interactions[col_ligand_receptor_pair].str.split(ligand_receptor_sep, expand=True)
            valid_mask = (filtered_interactions[col_ligand].isin(specificity_df.index)) & \
                         (filtered_interactions[col_receptor].isin(specificity_df.index))
            filtered_interactions = filtered_interactions[valid_mask]
            if filtered_interactions.empty: continue
            
            for _, row in filtered_interactions.iterrows():
                ligand, receptor = row[col_ligand], row[col_receptor]
                ligand_score = specificity_df.loc[ligand, sender_cell_type]
                receptor_score = specificity_df.loc[receptor, receiver_cell_type]
                global_vmax = max(global_vmax, ligand_score, receptor_score)

    # Store handles for shared legend/colorbar
    legend_handles, legend_labels, mappable = None, None, None
    
    for idx, cell_type_pair in enumerate(cell_type_pairs):
        # --- Left Column: Plotting Area ---
        plot_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=main_gs[idx, 0], 
                                                     height_ratios=[8, heatmap_height_ratio], hspace=0.1)
        ax_bar = fig.add_subplot(plot_gs[0])
        ax_hm = fig.add_subplot(plot_gs[1])

        # --- Data Preparation ---
        sender_cell_type, receiver_cell_type = cell_type_pair.split(cell_type_sep)
        top_interactions = interactions_df[interactions_df[col_cell_type_pair] == cell_type_pair].nlargest(top_n, col_interaction_score)
        if top_interactions.empty: continue

        filtered_interactions = top_interactions.copy()
        filtered_interactions[[col_ligand, col_receptor]] = filtered_interactions[col_ligand_receptor_pair].str.split(ligand_receptor_sep, expand=True)
        valid_mask = (filtered_interactions[col_ligand].isin(specificity_df.index)) & \
                     (filtered_interactions[col_receptor].isin(specificity_df.index))
        filtered_interactions = filtered_interactions[valid_mask]
        if filtered_interactions.empty: continue

        heatmap_scores, valid_ligand_receptors = [], []
        for _, row in filtered_interactions.iterrows():
            ligand, receptor = row[col_ligand], row[col_receptor]
            ligand_score = specificity_df.loc[ligand, sender_cell_type]
            receptor_score = specificity_df.loc[receptor, receiver_cell_type]
            heatmap_scores.append([ligand_score, receptor_score])
            valid_ligand_receptors.append(row[col_ligand_receptor_pair])
            
        heatmap_data = pd.DataFrame(np.array(heatmap_scores).T, index=['Ligand', 'Receptor'], columns=valid_ligand_receptors)
        
        # --- 1. Bar Plot ---
        N = len(filtered_interactions)
        bar_colors = filtered_interactions[col_annotation].map(color_palette)
        ax_bar.bar(
            x=np.arange(N),
            height=filtered_interactions[col_interaction_score],
            color=bar_colors,
            width=1.0, # Make bars touch each other
            edgecolor='black', # Add border to match heatmap
            linewidth=0.5
        )
        
        ax_bar.set_ylim(0, y_max)
        # Set x-axis limits to match the heatmap extent, removing padding
        ax_bar.set_xlim(-0.5, N - 0.5)
        
        ax_bar.set_title(f'Top {len(filtered_interactions)} Interactions for {cell_type_pair}', fontsize=18, pad=15)
        ax_bar.set_ylabel('Interaction Score', fontsize=14)
        ax_bar.set_xlabel('')
        ax_bar.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        # Create handles for the legend manually
        current_handles = [mpatches.Patch(color=color, label=label) for label, color in color_palette.items() if label in filtered_interactions[col_annotation].unique()]
        current_labels = [h.get_label() for h in current_handles]

        for patch in ax_bar.patches:
            if patch.get_height() > y_max:
                x_pos = patch.get_x() + patch.get_width() / 2
                ax_bar.text(x_pos, y_max * 0.95, '↑', ha='center', va='top', fontsize=12, fontweight='bold', color='red')

        # --- 2. Specificity Heatmap ---
        vmax_actual = heatmap_vmax if heatmap_vmax is not None else (global_vmax if shared_legend else heatmap_data.max().max())
        
        img = ax_hm.imshow(heatmap_data, cmap=heatmap_cmap, aspect='auto', interpolation='none',
                           extent=[-0.5, N - 0.5, 1.5, -0.5], vmin=0, vmax=vmax_actual)

        ax_hm.set_ylabel('')
        ax_hm.set_xlabel(f'Ligand{ligand_receptor_sep}Receptor Interaction', fontsize=14)
        ax_hm.set_yticks([0, 1])
        ax_hm.set_yticklabels(['Ligand', 'Receptor'], rotation=0, va='center', fontsize=12)
        ax_hm.set_xticks(np.arange(N))
        ax_hm.set_xticklabels(valid_ligand_receptors, rotation=45, ha='right', rotation_mode='anchor')
        
        ax_hm.set_xticks(np.arange(-0.5, N, 1), minor=True)
        ax_hm.set_yticks(np.arange(-0.5, 2, 1), minor=True)
        ax_hm.grid(which='minor', color='black', linestyle='-', linewidth=1)
        ax_hm.tick_params(which='minor', size=0)

        # --- Right Column: Legend and Colorbar Area ---
        if not shared_legend:
            right_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=main_gs[idx, 1], height_ratios=[2, 0.25], hspace=0.5)
            legend_ax = fig.add_subplot(right_gs[0])
            cbar_ax = fig.add_subplot(right_gs[1])
            
            legend_ax.legend(current_handles, current_labels, title='Annotation', loc='center left')
            legend_ax.axis('off')

            cbar = fig.colorbar(img, cax=cbar_ax, orientation='horizontal')
            cbar.set_label('Specificity Score', size=12)
            cbar.set_ticks([0, vmax_actual])
            cbar.set_ticklabels([f'0.00', f'{vmax_actual:.2f}'])
        else:
            if legend_handles is None:
                legend_handles, legend_labels = current_handles, current_labels
            if mappable is None:
                mappable = img

    # --- Add Shared Legend and Colorbar if requested ---
    if shared_legend:
        right_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=main_gs[0, 1], height_ratios=[2, 0.25], hspace=0.5)
        
        if legend_handles:
            legend_ax = fig.add_subplot(right_gs[0])
            # Use all_annotations to create the full legend
            fresh_handles = [mpatches.Patch(color=color_palette[label], label=label) for label in all_annotations]
            legend_ax.legend(handles=fresh_handles, title='Annotation', loc='center left', frameon=False)
            legend_ax.axis('off')
        if mappable:
            cbar_ax = fig.add_subplot(right_gs[1])
            cbar = fig.colorbar(mappable, cax=cbar_ax, orientation='horizontal')
            cbar.set_label('Specificity Score', size=12, labelpad=5)
            cbar.set_ticks([0, global_vmax])
            cbar.set_ticklabels([f'0.00', f'{global_vmax:.2f}'])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
