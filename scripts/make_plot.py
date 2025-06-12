"""
Standalone script to generate the colony attendance plot.
"""

import os
from pathlib import Path
import pandas as pd
from plotnine import (
    ggplot, aes, geom_rect, scale_x_datetime, scale_y_continuous,
    scale_fill_manual, labs, theme_bw, theme, element_text,
    element_blank, facet_wrap, coord_cartesian
)

def standardize_classes(df):
    """Standardize class names to match the visualization categories."""
    df = df.copy()
    
    # Map class names to standardized categories
    class_mapping = {
        'Few_Half': '<50%',
        'Many_All': '>50%',
        'Zero': 'Zero'
    }
    
    df['class_name'] = df['top_class'].map(class_mapping)
    # Create categorical class index for proper ordering in plot
    df['class_name'] = pd.Categorical(
        df['class_name'],
        categories=['Zero', '<50%', '>50%'],
        ordered=True
    )
    return df

def normalize_dates(df):
    """Normalize dates to a common year (2000) for comparison."""
    df = df.copy()
    df['year'] = pd.to_datetime(df['datetime_taken']).dt.year.astype(str)
    df['normalized_date'] = pd.to_datetime(df['datetime_taken']).apply(
        lambda x: x.replace(year=2000)
    )
    return df

def main():
    # Ensure figs directory exists
    os.makedirs('figs', exist_ok=True)
    
    # Load all available results data
    results_dir = Path('results')
    results_files = list(results_dir.glob('*results*.csv'))
    
    if not results_files:
        print("No results files found in the 'results' directory.")
        return
        
    dfs = []
    for f in results_files:
        df_temp = pd.read_csv(f, low_memory=False)
        df_temp['datetime_taken'] = pd.to_datetime(df_temp['datetime_taken'])
        dfs.append(df_temp)

    # Combine and prepare data
    df = pd.concat(dfs, ignore_index=True)
    df = standardize_classes(df)
    df = normalize_dates(df)
    df = df.dropna(subset=['datetime_taken', 'class_name'])
    
    # Define rectangle boundaries for each 30-second interval
    df['xmax'] = df['normalized_date'] + pd.to_timedelta(30, 's')
    
    # Get the number of years for dynamic plot sizing
    n_years = df['year'].nunique()
    plot_height = n_years * 2.5  # 2.5 inches per year
    
    # Define colors for the viridis palette
    colors = {
        'Zero': '#440154',  # Dark purple
        '<50%': '#21908C',  # Teal
        '>50%': '#FDE725'   # Yellow
    }
    
    # Create the plot
    plot = (
        ggplot(df, aes(fill='class_name'))
        + geom_rect(aes(xmin='normalized_date', xmax='xmax', ymin=0, ymax=1))
        + scale_x_datetime(date_labels='%m/%d', 
                         date_breaks='10 days',
                         expand=(0, 0))
        + scale_y_continuous(expand=(0, 0))
        + coord_cartesian(ylim=(0, 1))  # Enforce y limits
        + scale_fill_manual(values=colors, name='Image Classification')
        + labs(x='Date', 
              y='',  # Remove y-axis label
              title='Gull Island Common Murre Attendance',
              subtitle='preliminary classification results from Marker Camera')
        + theme_bw()
        + theme(
            axis_title_y=element_blank(),
            axis_text_y=element_blank(),
            axis_ticks_y=element_blank(),
            axis_title_x=element_text(margin={'t': 10}),  # Add margin to x-axis title
            legend_position='bottom',
            #plot_title=element_text(hjust=0.5),
            strip_background=element_blank(),
            strip_text_y=element_text(angle=0),  # Make year labels horizontal
            panel_grid_major_y=element_blank(),
            panel_spacing=0.01  # Add a little space between facets
        )
        + facet_wrap('~ year', ncol=1)  # Facet by year, stacked vertically
    )
    
    # Save the plot
    plot.save('figs/colony_attendance.png', dpi=300, width=10, height=plot_height, units='in')
    print("Plot has been saved to figs/colony_attendance.png")

if __name__ == "__main__":
    main() 