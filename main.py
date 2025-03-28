import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import tunnel data
# TUNEL A Data
tunnel_a_data = {
    "11:35": {
        "122678596": {
             "Lado Izquierdo": {
                  "ARRIBA": [4.4, 5.2, 5.8],
                  "MEDIO": [4.5, 5.6, 5.1],
                  "ABAJO": [5.6, 4.9, 4.3]
             }
        },
        "122678790": {
             "Lado Izquierdo": {
                  "ARRIBA": [7.3, 9.3, 7.0],
                  "MEDIO": [7.3, 8.7, 9.1],
                  "ABAJO": [6.1, 7.3, 6.4]
             }
        },
        "122678722": {
             "Lado Izquierdo": {
                  "ARRIBA": [7.5, 8.4, 6.5],
                  "MEDIO": [8.5, 9.8, 6.9],
                  "ABAJO": [6.7, 7.2, 6.0]
             }
        }
    },
    "12:56": {
        "122678596": {
             "Lado Izquierdo": {
                  "ARRIBA": [3.8, 5.1, 4.8],
                  "MEDIO": [4.0, 4.3, 3.7],
                  "ABAJO": [4.2, 4.1, 4.0]
             }
        },
        "122678790": {
             "Lado Izquierdo": {
                  "ARRIBA": [8.7, 7.6, 6.7],
                  "MEDIO": [5.1, 8.7, 4.2],
                  "ABAJO": [4.8, 3.7, 5.0]
             }
        },
        "122678722": {
             "Lado Izquierdo": {
                  "ARRIBA": [3.9, 4.7, None],
                  "MEDIO": [4.9, 4.6, None],
                  "ABAJO": [4.0, 3.9, None]
             }
        }
    },
    "14:04": {
        "122678596": {
             "Lado Izquierdo": {
                  "ARRIBA": [3.4, 4.0, 3.8],
                  "MEDIO": [3.4, 3.8, 3.8],
                  "ABAJO": [3.7, 3.9, 5.1]
             }
        },
        "122678790": {
             "Lado Izquierdo": {
                  "ARRIBA": [4.5, 6.1, 7.0],
                  "MEDIO": [5.2, 6.1, 5.0],
                  "ABAJO": [4.9, 5.6, 4.8]
             }
        },
        "122678722": {
             "Lado Izquierdo": {
                  "ARRIBA": [7.5, 8.4, 6.5],
                  "MEDIO": [8.5, 9.8, 6.9],
                  "ABAJO": [6.7, 7.2, 6.0]
             }
        }
    },
    "16:20": {
        "122678596": {
             "Lado Izquierdo": {
                  "ARRIBA": [3.2, 3.5, 3.3],
                  "MEDIO": [3.3, 3.7, 3.6],
                  "ABAJO": [4.6, 3.9, 3.3]
             }
        },
        "122678790": {
             "Lado Izquierdo": {
                  "ARRIBA": [3.9, 5.5, 4.0],
                  "MEDIO": [4.4, 5.4, 4.5],
                  "ABAJO": [3.9, 4.6, 3.9]
             }
        },
        "122678722": {
             "Lado Izquierdo": {
                  "ARRIBA": [4.5, 4.9, 4.1],
                  "MEDIO": [5.7, 6.0, 4.8],
                  "ABAJO": [4.0, 4.6, 4.2]
             }
        }
    }
}

# TUNEL F Data
tunnel_f_data = {
    "8:55": {
        "122677490": {
             "Lado Izquierdo": {
                  "ARRIBA": [16.8, 16.5, 16.6],
                  "MEDIO": [17.0, 16.5, 16.9],
                  "ABAJO": [16.0, 15.5, 15.6]
             },
             "Lado Derecho": {
                  "ARRIBA": [12.4, 12.6, 10.3],
                  "MEDIO": [12.8, 12.6, 8.3],
                  "ABAJO": [9.7, 10.3, 7.0]
             }
        },
        "122677496": {
             "Lado Izquierdo": {
                  "ARRIBA": [16.0, 16.3, 16.4],
                  "MEDIO": [16.6, 15.9, 16.3],
                  "ABAJO": [15.2, 15.6, 15.2]
             },
             "Lado Derecho": {
                  "ARRIBA": [13.3, 11.4, 12.7],
                  "MEDIO": [12.1, 16.3, 13.7],
                  "ABAJO": [9.5, 11.8, 11.2]
             }
        },
        "122677449": {
             "Lado Izquierdo": {
                  "ARRIBA": [21.3, 19.1, 18.5],
                  "MEDIO": [20.3, 20.3, 16.9],
                  "ABAJO": [16.6, 17.4, 14.7]
             },
             "Lado Derecho": {
                  "ARRIBA": [15.5, 14.3, 15.0],
                  "MEDIO": [15.1, 17.2, 13.5],
                  "ABAJO": [13.9, 11.9, 12.3]
             }
        }
    },
    "12:10": {
        "122677490": {
             "Lado Izquierdo": {
                  "ARRIBA": [5.8, 4.6, 6.8],
                  "MEDIO": [6.6, 7.9, 4.7],
                  "ABAJO": [6.9, 4.3, 5.1]
             },
             "Lado Derecho": {
                  "ARRIBA": [4.5, 6.3, 6.2],
                  "MEDIO": [4.8, 5.0, 6.8],
                  "ABAJO": [4.6, 5.5, 3.8]
             }
        },
        "122677496": {
             "Lado Izquierdo": {
                  "ARRIBA": [5.8, 11.2, 5.6],
                  "MEDIO": [6.7, 8.2, 4.6],
                  "ABAJO": [5.9, 6.2, 6.4]
             },
             "Lado Derecho": {
                  "ARRIBA": [8.8, 6.4, 4.5],
                  "MEDIO": [6.1, 10.6, 9.7],
                  "ABAJO": [5.7, 7.4, 6.6]
             }
        },
        "122677449": {
             "Lado Izquierdo": {
                  "ARRIBA": [7.0, 9.3, 5.9],
                  "MEDIO": [9.6, 9.2, 6.5],
                  "ABAJO": [6.7, 6.9, 5.9]
             },
             "Lado Derecho": {
                  "ARRIBA": [4.9, 7.0, 6.1],
                  "MEDIO": [8.8, 5.8, 5.0],
                  "ABAJO": [6.9, 5.0, 4.5]
             }
        }
    }
}

# TUNEL E Data
tunnel_e_data = {
    "08:55 Exterior": {
        "122677503": {
             "Lado Izquierdo": {
                  "ARRIBA": [14.1, 10.3, 14.4],
                  "MEDIO": [13.5, 12.0, 12.8],
                  "ABAJO": [11.2, 9.7, 11.2]
             }
        },
        "122677446": {
             "Lado Izquierdo": {
                  "ARRIBA": [9.6, 8.1, 8.8],
                  "MEDIO": [10.1, 9.1, 9.2],
                  "ABAJO": [9.1, 8.5, 10.2]
             }
        },
        "122677394": {
             "Lado Izquierdo": {
                  "ARRIBA": [8.3, 8.3, 9.7],
                  "MEDIO": [10.4, 8.6, 9.5],
                  "ABAJO": [9.9, 9.2, 10.3]
             }
        }
    },
    "08:55 Interior": {
        "122677503": {
             "Lado Izquierdo": {
                  "ARRIBA": [11.9, 9.9, None],
                  "MEDIO": [9.9, 8.3, None],
                  "ABAJO": [7.5, 9.4, None]
             }
        },
        "122677446": {
             "Lado Izquierdo": {
                  "ARRIBA": [10.6, 10.8, 10.7],
                  "MEDIO": [9.6, 9.6, 11.7],
                  "ABAJO": [8.4, 8.0, 8.4]
             }
        },
        "122677394": {
             "Lado Izquierdo": {
                  "ARRIBA": [10.3, 9.6, 10.9],
                  "MEDIO": [9.5, 9.3, 9.7],
                  "ABAJO": [8.3, 7.2, 8.0]
             }
        }
    },
    "11:55 Segundo Piso": {
        "122677477": {
             "Lado Izquierdo": {
                  "ARRIBA": [9.9, 8.0, 10.4],
                  "MEDIO": [6.8, 7.5, 8.0],
                  "ABAJO": [7.6, 7.2, 7.7]
             }
        },
        "122677412": {
             "Lado Izquierdo": {
                  "ARRIBA": [6.7, 6.5, 7.6],
                  "MEDIO": [8.8, 8.9, 8.8],
                  "ABAJO": [7.1, 7.6, 7.4]
             }
        },
        "122677410": {
             "Lado Izquierdo": {
                  "ARRIBA": [7.8, 9.9, 10.5],
                  "MEDIO": [9.1, 10.3, 9.3],
                  "ABAJO": [7.5, 7.6, 7.2]
             }
        }
    },
    "12:51 Exterior": {
        "122677503": {
             "Lado Izquierdo": {
                  "ARRIBA": [10.4, 7.1, 7.0],
                  "MEDIO": [8.3, 9.1, 8.3],
                  "ABAJO": [8.0, 5.8, 8.5]
             }
        },
        "122677446": {
             "Lado Izquierdo": {
                  "ARRIBA": [6.0, 5.3, 6.2],
                  "MEDIO": [6.3, 5.6, 5.9],
                  "ABAJO": [6.4, 5.7, 6.8]
             }
        },
        "122677394": {
             "Lado Izquierdo": {
                  "ARRIBA": [6.1, 5.8, 6.9],
                  "MEDIO": [6.4, 5.4, 6.2],
                  "ABAJO": [6.7, 6.1, 6.3]
             }
        }
    },
    "15:20 Exterior y Segundo Piso": {
        "122677503": {
             "Lado Izquierdo": {
                  "ARRIBA": [7.1, 6.4, 6.1],
                  "MEDIO": [7.1, 6.4, 6.1],
                  "ABAJO": [7.0, 4.8, 6.9]
             }
        },
        "122677446": {
             "Lado Izquierdo": {
                  "ARRIBA": [4.9, 4.9, 5.2],
                  "MEDIO": [4.0, 4.9, 5.2],
                  "ABAJO": [5.3, 4.6, 5.9]
             }
        },
        "122677394": {
             "Lado Izquierdo": {
                  "ARRIBA": [4.6, 4.4, 5.3],
                  "MEDIO": [4.9, 4.4, 4.7],
                  "ABAJO": [5.5, 4.8, 4.7]
             }
        },
        "122677477": {
             "Lado Izquierdo": {
                  "ARRIBA": [4.0, 4.2, 4.3],
                  "MEDIO": [3.8, 3.8, 4.1],
                  "ABAJO": [4.2, 3.5, 3.8]
             }
        },
        "122677412": {
             "Lado Izquierdo": {
                  "ARRIBA": [3.8, 3.6, 3.8],
                  "MEDIO": [3.9, 4.2, 4.3],
                  "ABAJO": [3.4, 3.5, 3.4]
             }
        },
        "122677410": {
             "Lado Izquierdo": {
                  "ARRIBA": [4.4, 6.4, 4.6],
                  "MEDIO": [5.0, 4.3, 5.0],
                  "ABAJO": [3.4, 3.8, 3.4]
             }
        }
    },
    "15:40 Interior": {
        "122677503": {
             "Lado Izquierdo": {
                  "ARRIBA": [6.4, 6.7, None],
                  "MEDIO": [5.8, 6.6, None],
                  "ABAJO": [4.0, 4.3, None]
             }
        },
        "122677446": {
             "Lado Izquierdo": {
                  "ARRIBA": [6.0, 5.9, 6.8],
                  "MEDIO": [6.0, 6.3, 7.3],
                  "ABAJO": [4.8, 4.5, 5.1]
             }
        },
        "122677394": {
             "Lado Izquierdo": {
                  "ARRIBA": [6.2, 5.8, 6.5],
                  "MEDIO": [6.3, 5.4, 6.0],
                  "ABAJO": [4.6, 4.4, 4.2]
             }
        }
    }
}

# TUNEL G Data
tunnel_g_data = {
    "13:50 Exterior": {
        "122679598": {
             "Lado Izquierdo": {
                  "ARRIBA": [16.0, 16.0, 16.7],
                  "MEDIO": [17.1, 17.5, 16.7],
                  "ABAJO": [16.1, 18.0, 16.8]
             }
        },
        "122679586": {
             "Lado Izquierdo": {
                  "ARRIBA": [16.1, 16.4, 16.9],
                  "MEDIO": [16.7, 17.1, 17.0],
                  "ABAJO": [16.3, 17.7, 17.0]
             }
        },
        "122679580": {
             "Lado Izquierdo": {
                  "ARRIBA": [16.9, 16.8, 16.2],
                  "MEDIO": [17.1, 16.7, 16.4],
                  "ABAJO": [16.9, 17.3, 17.0]
             }
        }
    },
    "13:50 Interior": {
        "122679598": {
             "Lado Izquierdo": {
                  "ARRIBA": [16.3, 17.6, 16.6],
                  "MEDIO": [17.4, 16.3, 16.9],
                  "ABAJO": [16.0, 16.7, 16.9]
             }
        },
        "122679586": {
             "Lado Izquierdo": {
                  "ARRIBA": [16.6, 17.2, 16.8],
                  "MEDIO": [16.5, 16.8, 17.0],
                  "ABAJO": [16.5, 17.0, 16.7]
             }
        },
        "122679580": {
             "Lado Izquierdo": {
                  "ARRIBA": [16.7, 16.5, 16.9],
                  "MEDIO": [16.8, 16.3, 16.2],
                  "ABAJO": [16.7, 16.6, 16.3]
             }
        }
    },
    "16:50 Exterior": {
        "122679598": {
             "Lado Izquierdo": {
                  "ARRIBA": [4.4, 4.5, 4.7],
                  "MEDIO": [5.5, 5.3, 5.1],
                  "ABAJO": [6.0, 6.2, 6.1]
             }
        },
        "122679586": {
             "Lado Izquierdo": {
                  "ARRIBA": [4.8, 4.5, 4.6],
                  "MEDIO": [5.3, 5.9, 5.6],
                  "ABAJO": [5.9, 6.4, 6.0]
             }
        },
        "122679580": {
             "Lado Izquierdo": {
                  "ARRIBA": [4.6, 4.4, 4.7],
                  "MEDIO": [6.8, 6.6, 6.6],
                  "ABAJO": [7.4, 7.0, 7.0]
             }
        }
    },
    "16:50 Interior": {
        "122679598": {
             "Lado Izquierdo": {
                  "ARRIBA": [4.1, 4.0, 4.1],
                  "MEDIO": [5.6, 5.4, 5.8],
                  "ABAJO": [4.9, 5.0, 4.8]
             }
        },
        "122679586": {
             "Lado Izquierdo": {
                  "ARRIBA": [4.2, 4.3, 4.0],
                  "MEDIO": [6.0, 5.8, 5.7],
                  "ABAJO": [4.9, 4.7, 4.8]
             }
        },
        "122679580": {
             "Lado Izquierdo": {
                  "ARRIBA": [4.6, 4.5, 4.5],
                  "MEDIO": [6.6, 6.2, 6.3],
                  "ABAJO": [5.1, 4.9, 4.8]
             }
        }
    }
}

# Consolidate all tunnel data
all_tunnels = {
    "Túnel A": tunnel_a_data,
    "Túnel E": tunnel_e_data,
    "Túnel F": tunnel_f_data,
    "Túnel G": tunnel_g_data
}

def generate_heatmap(data, folio, group_name, row_order, x_labels):
    """Generate a heatmap matrix for a specific folio and group."""
    positions = data[folio][group_name]
    # Handle None values and convert to a proper numpy array
    matrix = []
    for row in row_order:
        row_data = positions[row]
        # Replace None with np.nan for proper heatmap rendering
        row_values = [x if x is not None else np.nan for x in row_data]
        matrix.append(row_values)
    
    return np.array(matrix)

def calculate_statistics(data):
    """Calculate statistics from the temperature data."""
    all_temps = []
    for folio in data:
        for group in data[folio]:
            for position in data[folio][group]:
                temps = [t for t in data[folio][group][position] if t is not None]
                all_temps.extend(temps)
    
    if not all_temps:
        return {"min": 0, "max": 0, "avg": 0, "median": 0}
    
    return {
        "min": np.min(all_temps),
        "max": np.max(all_temps),
        "avg": np.mean(all_temps),
        "median": np.median(all_temps)
    }

def interactive_heatmaps(data, measurement_time):
    """
    Generate interactive heatmaps for each folio in the data.
    """
    if not data:
        st.warning("No hay datos disponibles para esta selección.")
        return
    
    # Calculate statistics for the current dataset
    stats = calculate_statistics(data)
    
    # Display statistics in a formatted way
    st.subheader(f"Estadísticas de Temperatura - {measurement_time}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Temperatura Mínima", f"{stats['min']:.1f}°C")
    col2.metric("Temperatura Máxima", f"{stats['max']:.1f}°C")
    col3.metric("Temperatura Promedio", f"{stats['avg']:.1f}°C")
    col4.metric("Temperatura Mediana", f"{stats['median']:.1f}°C")
    
    row_order = ["ARRIBA", "MEDIO", "ABAJO"]
    x_labels = [f"Medición {i+1}" for i in range(3)]
    
    for folio, groups in data.items():
        st.subheader(f"Folio {folio}")
        n_groups = len(groups)
        
        # Create a figure with subplots (one row, n_groups columns)
        fig = make_subplots(
            rows=1, 
            cols=n_groups, 
            subplot_titles=[group for group in groups.keys()],
            horizontal_spacing=0.1
        )
        
        col = 1
        for group_name in groups.keys():
            # Generate the heatmap matrix
            matrix = generate_heatmap(data, folio, group_name, row_order, x_labels)
            
            # Create the heatmap trace
            heatmap = go.Heatmap(
                z=matrix,
                x=x_labels,
                y=row_order,
                colorscale='Viridis',
                hovertemplate="Posición: %{y}<br>Medición: %{x}<br>Temperatura: %{z}°C<extra></extra>",
                colorbar=dict(title="Temp. °C") if col == n_groups else None
            )
            
            fig.add_trace(heatmap, row=1, col=col)
            col += 1
        
        # Update layout for better visualization
        fig.update_layout(
            height=300,
            width=max(400, 200 * n_groups),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Main Streamlit app
def main():
    st.set_page_config(
        page_title="Visualización de Túneles de Prefrío",
        page_icon="🧊",
        layout="wide"
    )
    
    st.title("Visualización de Temperaturas en Túneles de Prefrío")
    st.markdown("""
    Esta aplicación permite visualizar las mediciones de temperatura en diferentes túneles de prefrío.
    Seleccione el túnel y la medición que desea visualizar.
    """)
    
    # Sidebar for tunnel selection
    with st.sidebar:
        st.header("Opciones de Visualización")
        
        # Tunnel selection (multi-select)
        selected_tunnels = st.multiselect(
            "Seleccionar Túneles:",
            options=list(all_tunnels.keys()),
            default=["Túnel A"]
        )
        
        # Dynamically create measurement options based on selected tunnels
        all_measurements = set()
        for tunnel in selected_tunnels:
            all_measurements.update(all_tunnels[tunnel].keys())
        
        # Sort measurements by time
        sorted_measurements = sorted(list(all_measurements))
        
        # Measurement selection
        selected_measurements = st.multiselect(
            "Seleccionar Mediciones:",
            options=sorted_measurements,
            default=sorted_measurements[:1] if sorted_measurements else []
        )
        
        # Option to show/hide stats
        show_stats = st.checkbox("Mostrar Estadísticas", value=True)
    
    # Main content area
    if not selected_tunnels:
        st.warning("Por favor, seleccione al menos un túnel para visualizar.")
    elif not selected_measurements:
        st.warning("Por favor, seleccione al menos una medición para visualizar.")
    else:
        # Display selected tunnels and measurements
        for tunnel in selected_tunnels:
            st.header(f"{tunnel}")
            
            for measurement in selected_measurements:
                if measurement in all_tunnels[tunnel]:
                    st.subheader(f"Medición: {measurement}")
                    interactive_heatmaps(all_tunnels[tunnel][measurement], measurement_time=measurement)
                    st.markdown("---")

if __name__ == "__main__":
    main()