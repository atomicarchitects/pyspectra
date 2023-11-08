from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

@app.route('/')
def scatterplot():
    # Create a sample dataframe for scatter plot
    data = {'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 1, 3, 5]}
    df = pd.DataFrame(data)

    # Create a list of sample dataframes for surface plots
    surface_data_list = [
        pd.DataFrame({'x_values': [1, 2, 3, 4], 'y_values': [10, 15, 13, 17], 'z_values': [12, 9, 15, 12], 'color_values': [9, 17, 12, 11]}),
        pd.DataFrame({'x_values': [3, 1, 2, 4], 'y_values': [12, 8, 10, 18], 'z_values': [10, 11, 14, 13], 'color_values': [7, 16, 15, 9]}),
        pd.DataFrame({'x_values': [2, 3, 1, 4], 'y_values': [11, 14, 12, 16], 'z_values': [13, 10, 16, 11], 'color_values': [8, 15, 14, 10]}),
        pd.DataFrame({'x_values': [4, 1, 3, 2], 'y_values': [13, 9, 11, 17], 'z_values': [11, 12, 13, 14], 'color_values': [6, 18, 16, 8]}),
        pd.DataFrame({'x_values': [1, 4, 2, 3], 'y_values': [10, 16, 11, 18], 'z_values': [12, 13, 14, 15], 'color_values': [9, 17, 16, 10]})
    ]
    
    surface_data_list = [df.to_dict() for df in surface_data_list]

    return render_template('scatterplot.html', scatter_data=df.to_dict(), surface_data_list=surface_data_list)

if __name__ == '__main__':
    app.run(debug=True)
    