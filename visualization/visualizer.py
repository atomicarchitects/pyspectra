import sys
sys.path.append('../src/')

from flask import Flask, render_template
import pandas as pd
import spectra
import jax.numpy as jnp 

app = Flask(__name__)

@app.route('/')
def scatterplot():
    # Create a sample dataframe for scatter plot
    data = {'x': [1, 2],
            'y': [2, 4]}
    df = pd.DataFrame(data)

    true_geometry = jnp.asarray([
        [1, 0, 0],
        [-0.5, jnp.sqrt(3)/2, 0],
        [-0.5, -jnp.sqrt(3)/2, 0]
    ])
    surface_data_list = [spectra.visualize(true_geometry)] * 2

    
    surface_data_list = [fig.to_json() for fig in surface_data_list]

    return render_template('scatterplot.html', scatter_data=df.to_dict(), surface_data_list=surface_data_list)

if __name__ == '__main__':
    app.run(debug=True)
    