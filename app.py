from shiny import App, ui, render, reactive
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define a consistent color palette
COLORS = {
    'primary': '#3498db',    # blue
    'secondary': '#e74c3c',  # red
    'tertiary': '#2ecc71',   # green
    'quaternary': '#f39c12', # orange
    'background': '#f5f5f5', # light gray
    'text': '#2c3e50',       # dark blue/gray
}

app_ui = ui.page_fluid(
    ui.tags.style(
        """
        body {
            background-color: #f5f5f5;
            padding: 20px;
            color: #2c3e50;
        }
        .app-title {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .app-author {
            font-size: 14px;
            font-style: italic;
            margin-bottom: 20px;
            color: #7f8c8d;
        }
        .section-title {
            background-color: #eaeaea;
            padding: 5px 10px;
            border-left: 4px solid #3498db;
        }
        .file-size-info {
            margin-top: 5px;
        }
        """
    ),
    ui.div(ui.h2("Let Shiny Do EDA For You", class_="app-title"),
           ui.p("By Akshay Aloriya", class_="app-author")),
    
    ui.input_file("file", "Choose CSV File (Max size: 15 MB)", accept=".csv", multiple=False),
    ui.output_text_verbatim("file_info"),
    ui.output_data_frame("file_preview"),
    
    ui.br(),
    ui.h3("Summary Statistics", class_="section-title"),
    ui.output_data_frame("summary_stats"),
    
    ui.br(),
    ui.h3("Missing Values Analysis", class_="section-title"),
    ui.output_data_frame("missing_values"),
    
    ui.br(),
    ui.h3("Column Visualization and Filtering (Numerical)", class_="section-title"),
    ui.input_select("num_column", "Select Numerical Column", choices=[]),
    ui.input_slider("range_filter", "Filter Range", min=0, max=100, value=[0, 100]),
    ui.output_plot("histogram"),
    ui.output_plot("boxplot"),
    
    ui.br(), 
    ui.h3("Categorical Column Visualization", class_="section-title"),
    ui.input_select("cat_column", "Select Categorical Column (≤8 categories)", choices=[]),
    ui.output_plot("barplot"),
    
    ui.br(), 
    ui.h3("Scatter Plot (Two Numerical Columns)", class_="section-title"),
    ui.input_select("scatter_x", "Select X-axis Numerical Column", choices=[]),
    ui.input_select("scatter_y", "Select Y-axis Numerical Column", choices=[]),
    ui.input_select("scatter_color", "Color by Categorical Column (optional)", choices=["None"], selected="None"),
    ui.output_plot("scatterplot"),
    
    ui.br(), 
    ui.output_text_verbatim("debug_info")
)

def server(input, output, session):
    @reactive.Calc
    def uploaded_data():
        file = input.file()
        if not file:
            return None
        try:
            # Check file size (15 MB limit)
            file_size_mb = file[0]["size"] / (1024 * 1024)
            if file_size_mb > 15:
                print(f"File too large: {file_size_mb:.2f} MB")
                return None
                
            df = pd.read_csv(file[0]["datapath"])
            print(f"Data loaded: {df.shape}")
            return df
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return None

    @reactive.Effect
    @reactive.event(input.file)
    def update_num_column_choices():
        data = uploaded_data()
        if data is not None and not data.empty:
            numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            continuous_cols = [col for col in numeric_cols if data[col].nunique() > 8]
            if continuous_cols:
                ui.update_select("num_column", choices=continuous_cols, selected=continuous_cols[0])
                ui.update_select("scatter_x", choices=continuous_cols, selected=continuous_cols[0])
                ui.update_select("scatter_y", choices=continuous_cols, selected=continuous_cols[1] if len(continuous_cols) > 1 else continuous_cols[0])
            else:
                ui.update_select("num_column", choices=[])
                ui.update_select("scatter_x", choices=[])
                ui.update_select("scatter_y", choices=[])

    @reactive.Effect
    @reactive.event(input.file)
    def update_cat_column_choices():
        data = uploaded_data()
        if data is not None and not data.empty:
            cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            discrete_numeric_cols = [col for col in numeric_cols if data[col].nunique() <= 8]
            all_cat_cols = cat_cols + discrete_numeric_cols
            valid_cat_cols = [col for col in all_cat_cols if data[col].nunique() <= 8]
            if valid_cat_cols:
                ui.update_select("cat_column", choices=valid_cat_cols, selected=valid_cat_cols[0])
                ui.update_select("scatter_color", choices=["None"] + valid_cat_cols, selected="None")
            else:
                ui.update_select("cat_column", choices=[])
                ui.update_select("scatter_color", choices=["None"])

    @reactive.Effect
    @reactive.event(input.num_column)
    def update_slider_range():
        data = uploaded_data()
        selected_col = input.num_column()
        if data is not None and selected_col in data.columns:
            try:
                col_min = float(data[selected_col].min())
                col_max = float(data[selected_col].max())
                if pd.isna(col_min) or pd.isna(col_max):
                    col_min, col_max = 0.0, 100.0
                ui.update_slider("range_filter", min=col_min, max=col_max, value=[col_min, col_max])
            except Exception as e:
                print(f"Error updating slider: {e}")
                ui.update_slider("range_filter", min=0, max=100, value=[0, 100])

    @reactive.Calc
    def filtered_data():
        data = uploaded_data()
        selected_col = input.num_column()
        if data is None or selected_col is None or selected_col not in data.columns:
            return data
        try:
            range_vals = input.range_filter()
            filtered = data[data[selected_col].between(range_vals[0], range_vals[1], inclusive='both')]
            print(f"Filtered data shape: {filtered.shape}")
            return filtered
        except Exception as e:
            print(f"Error filtering data: {e}")
            return data

    @output
    @render.text
    def file_info():
        file = input.file()
        if not file:
            return "No file uploaded yet"
        
        # Show size in KB if less than 1 MB
        file_size_bytes = file[0]['size']
        if file_size_bytes < 1024 * 1024:  # Less than 1 MB
            file_size_kb = file_size_bytes / 1024
            size_str = f"{file_size_kb:.2f} KB"
        else:
            file_size_mb = file_size_bytes / (1024 * 1024)
            size_str = f"{file_size_mb:.2f} MB"
            
        return f"Uploaded file: {file[0]['name']}\nSize: {size_str}"

    @output
    @render.data_frame
    def file_preview():
        data = uploaded_data()
        if data is None:
            return render.DataGrid(pd.DataFrame(), height="300px")
        return render.DataGrid(data.head(), height="300px")

    @output
    @render.data_frame
    def summary_stats():
        data = filtered_data()
        if data is None:
            return render.DataGrid(pd.DataFrame(), height="300px")
        numeric_data = data.select_dtypes(include=['int64', 'float64'])
        if numeric_data.empty:
            return render.DataGrid(pd.DataFrame({"Note": ["No numerical columns found"]}))
        try:
            summary = numeric_data.describe().reset_index()
            return render.DataGrid(summary, height="300px")
        except Exception as e:
            print(f"Error in summary stats: {e}")
            return render.DataGrid(pd.DataFrame({"Error": [str(e)]}))

    @output
    @render.data_frame
    def missing_values():
        data = uploaded_data()
        if data is None:
            return render.DataGrid(pd.DataFrame(), height="300px")
        
        try:
            # Calculate missing values statistics
            missing_stats = pd.DataFrame({
                'Column': data.columns,
                'Data Type': data.dtypes.astype(str),
                'Missing Values': data.isna().sum(),
                'Missing (%)': (data.isna().sum() / len(data) * 100).round(2),
                'Complete Values': data.notna().sum(),
                'Complete (%)': (data.notna().sum() / len(data) * 100).round(2)
            })
            return render.DataGrid(missing_stats, height="300px")
        except Exception as e:
            print(f"Error in missing values stats: {e}")
            return render.DataGrid(pd.DataFrame({"Error": [str(e)]}))

    @output
    @render.plot
    def histogram():
        data = filtered_data()
        selected_col = input.num_column()
        if data is None or selected_col is None or selected_col not in data.columns:
            return None
        try:
            fig, ax = plt.subplots()
            ax.hist(data[selected_col].dropna(), bins=30, color=COLORS['primary'], edgecolor='black')
            ax.set_title(f"Distribution of {selected_col}")
            ax.set_xlabel(selected_col)
            ax.set_ylabel("Count")
            ax.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error creating histogram: {e}")
            return None

    @output
    @render.plot
    def boxplot():
        data = filtered_data()
        selected_col = input.num_column()
        if data is None or selected_col is None or selected_col not in data.columns:
            return None
        try:
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.boxplot(data[selected_col].dropna(), vert=False, patch_artist=True,
                      showfliers=True, boxprops=dict(facecolor=COLORS['primary']),
                      medianprops=dict(color=COLORS['secondary']), 
                      whiskerprops=dict(color=COLORS['text']),
                      flierprops=dict(marker='o', markerfacecolor=COLORS['secondary'], markersize=5))
            ax.set_title(f"Boxplot of {selected_col}")
            ax.set_xlabel(selected_col)
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error creating boxplot: {e}")
            return None

    @output
    @render.plot
    def barplot():
        data = uploaded_data()
        selected_col = input.cat_column()
        if data is None or selected_col is None or selected_col not in data.columns:
            return None
        try:
            fig, ax = plt.subplots(figsize=(8, 4))
            value_counts = data[selected_col].value_counts()
            ax.bar(value_counts.index, value_counts.values, color=COLORS['tertiary'], edgecolor='black')
            ax.set_title(f"Bar Plot of {selected_col}")
            ax.set_xlabel(selected_col)
            ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error creating barplot: {e}")
            return None

    @output
    @render.plot
    def scatterplot():
        data = filtered_data()
        x_col = input.scatter_x()
        y_col = input.scatter_y()
        color_col = input.scatter_color()
        
        if (data is None or x_col is None or y_col is None or 
            x_col not in data.columns or y_col not in data.columns):
            return None
        
        try:
            # Make figure wider to accommodate legend
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if color_col == "None" or color_col not in data.columns:
                ax.scatter(data[x_col], data[y_col], alpha=0.5, color=COLORS['quaternary'])
            else:
                categories = data[color_col].dropna().unique()
                # Fixed: using matplotlib.colormaps instead of get_cmap
                colormap = plt.colormaps['Set1']
                for i, category in enumerate(categories):
                    category_data = data[data[color_col] == category]
                    ax.scatter(category_data[x_col], 
                              category_data[y_col], 
                              alpha=0.5, 
                              color=colormap(i % colormap.N), 
                              label=str(category))
                # Place legend outside and to the right of the axes
                ax.legend(title=color_col, bbox_to_anchor=(1.02, 1), loc='upper left')
                # Adjust layout to make room for legend
                plt.subplots_adjust(right=0.8)

            ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.grid(True, linestyle='--', alpha=0.7)
            return fig
        except Exception as e:
            print(f"Error creating scatterplot: {e}")
            return None

    @output
    @render.text
    def debug_info():
        data = filtered_data()
        raw_data = uploaded_data()
        num_col = input.num_column()
        cat_col = input.cat_column()
        scatter_x = input.scatter_x()
        scatter_y = input.scatter_y()
        scatter_color = input.scatter_color()
        if raw_data is None:
            return "No data uploaded"
        if num_col is None:
            return "No numerical column selected"
        if data is None:
            return "Filtered data is None"
        return (f"Raw data shape: {raw_data.shape}\n"
                f"Filtered data shape: {data.shape}\n"
                f"Selected numerical column: {num_col}\n"
                f"Selected categorical column: {cat_col}\n"
                f"Scatter X column: {scatter_x}\n"
                f"Scatter Y column: {scatter_y}\n"
                f"Scatter color column: {scatter_color}\n"
                f"Num column type: {raw_data[num_col].dtype}\n"
                f"Cat column type: {raw_data[cat_col].dtype if cat_col in raw_data.columns else 'N/A'}\n"
                f"Filter range: {input.range_filter()}\n"
                f"Cat column unique values: {raw_data[cat_col].nunique() if cat_col in raw_data.columns else 'N/A'}")

app = App(app_ui, server)

if __name__ == "__main__":
    app.run()