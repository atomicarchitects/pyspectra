import plotly.graph_objects as go
import e3nn_jax as e3nn
from spectra import with_peaks_at
import jax.numpy as jnp


def visualize_signal(signal):
    layout = go.Layout(
        scene=dict(
            xaxis=dict(
                title='', 
                showticklabels=False, 
                showgrid=False, 
                zeroline=False, 
                backgroundcolor='rgba(255,255,255,255)', 
                range=[-2.5, 2.5]
            ),
            yaxis=dict(
                title='', 
                showticklabels=False, 
                showgrid=False, 
                zeroline=False, 
                backgroundcolor='rgba(255,255,255,255)', 
                range=[-2.5, 2.5]
            ),
            zaxis=dict(
                title='', 
                showticklabels=False, 
                showgrid=False, 
                zeroline=False, 
                backgroundcolor='rgba(255,255,255,255)', 
                range=[-2.5, 2.5]
            ),
            bgcolor='rgba(255,255,255,255)',
            aspectmode='cube',
            camera=dict(
                eye=dict(x=0.5, y=0.5, z=0.5)
            )
        ),
        plot_bgcolor='rgba(255,255,255,255)',
        paper_bgcolor='rgba(255,255,255,255)',
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            yanchor="top",
            y=1,
            xanchor="left",
            x=0
        )
    )

    spherical_harmonics_trace = go.Surface(
        e3nn.to_s2grid(signal, 100, 99, quadrature="soft").plotly_surface(
            radius=1., 
            normalize_radius_by_max_amplitude=True, 
            scale_radius_by_amplitude=True
        ), 
        name="Signal", 
        showlegend=True
    )

    fig = go.Figure()
    fig.add_trace(spherical_harmonics_trace)
    fig.update_layout(layout)
    return fig


def visualize_geometry(geometry, lmax=4, show_points=False, use_sum_of_diracs=False):
    signal = with_peaks_at(geometry, lmax=lmax, use_sum_of_diracs=use_sum_of_diracs)
    fig = visualize_signal(signal)
    if show_points:
        atoms_trace = go.Scatter3d(
            x=geometry[:, 0], 
            y=geometry[:, 1], 
            z=geometry[:, 2], 
            mode='markers', 
            marker=dict(size=10, color='black'), 
            showlegend=True, 
            name="Points"
        )
        fig.add_trace(atoms_trace)
    return fig


def colorplot(arr: jnp.ndarray):
    """Plot spectra"""
    # TODO: add functionality to plot multiple spectra (properly take into account max and min)
    # TODO: add functionality to change dimensions of figure
    reshaped_arr = arr.reshape(3, 5)  # Reshape the array into 3 rows of 5 components each
    plt.figure(figsize=(15, 3))  # Adjust the figure size to accommodate the reshaped array
    plt.axis("off")
    vmax = jnp.maximum(jnp.abs(jnp.min(reshaped_arr)), jnp.max(reshaped_arr))  # Compute vmax using the reshaped array
    return plt.imshow(reshaped_arr, cmap="PuOr", vmin=-vmax, vmax=vmax)  # Plot the reshaped array