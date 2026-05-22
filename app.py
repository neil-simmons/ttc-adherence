def generate_kepler_config():
    custom_20_colors = [
        "#DA251D", "#E03920", "#E54E23", "#EB6326", "#F07729", 
        "#F58C2C", "#FBA02F", "#FFB532", "#FFCA35", "#FFDE38", 
        "#F2E43B", "#D8DB3D", "#BED240", "#A3C942", "#89C045", 
        "#6FB747", "#54AE4A", "#3AA54C", "#209C4F", "#1A9641"
    ]
    color_scale_config = {"name": "TTC_Scale", "type": "custom", "category": "Custom", "colors": custom_20_colors}
    return {
        "version": "v1",
        "config": {
            "visState": {
                "layers": [
                    {
                        "id": "stops", "type": "point",
                        "config": {
                            "dataId": "stops", "label": "Stops", "columns": {"lat": "stop_lat", "lng": "stop_lon"}, "isVisible": True,
                            "visConfig": {
                                "radius": 4.5, # Small, crisp screen pixel radius (stays sharp at all zoom levels)
                                "radiusUnit": "pixels", 
                                "opacity": 0.95, 
                                "filled": True, 
                                "outline": True, 
                                "thickness": 0.8, # Thin outline matches the pinhead scale
                                "strokeColor": [220, 220, 220], 
                                "colorRange": color_scale_config
                            }
                        },
                        "visualChannels": {"colorField": {"name": "reliability", "type": "real"}, "colorScale": "quantize"}
                    },
                    {
                        "id": "segments", "type": "geojson",
                        "config": {
                            "dataId": "segments", "label": "Route Segments", "columns": {"geojson": "geometry"}, "isVisible": True,
                            "visConfig": {
                                "opacity": 0.5, 
                                "strokeOpacity": 0.5, 
                                "thickness": 0.6, 
                                "strokeColor": None, 
                                "colorRange": color_scale_config, 
                                "strokeColorRange": color_scale_config
                            }
                        },
                        "visualChannels": {"colorField": {"name": "avg_reliability", "type": "real"}, "colorScale": "quantize", "strokeColorField": {"name": "avg_reliability", "type": "real"}, "strokeColorScale": "quantize"}
                    }
                ],
                "layerOrder": ["stops", "segments"],
                "interactionConfig": {
                    "tooltip": {
                        "fieldsToShow": {
                            "segments": [{"name": "segment", "format": None}, {"name": "route_id", "format": None}, {"name": "avg_reliability", "format": ".1f"}],
                            "stops": [{"name": "stop_name", "format": None}, {"name": "route_id", "format": None}, {"name": "reliability", "format": ".1f"}]
                        },
                        "enabled": True
                    }
                }
            },
            "mapStyle": {"styleType": "muted_night"}
        }
    }

def generate_equity_kepler_config():
    custom_20_colors = [
        "#DA251D", "#E03920", "#E54E23", "#EB6326", "#F07729", 
        "#F58C2C", "#FBA02F", "#FFB532", "#FFCA35", "#FFDE38", 
        "#F2E43B", "#D8DB3D", "#BED240", "#A3C942", "#89C045", 
        "#6FB747", "#54AE4A", "#3AA54C", "#209C4F", "#1A9641"
    ]
    color_scale_config = {"name": "TTC_Scale", "type": "custom", "category": "Custom", "colors": custom_20_colors}
    
    return {
        "version": "v1",
        "config": {
            "visState": {
                "layers": [
                    {
                        "id": "stops", "type": "point",
                        "config": {
                            "dataId": "stops", "label": "Stops", "columns": {"lat": "stop_lat", "lng": "stop_lon"}, "isVisible": True,
                            "visConfig": {
                                "radius": 4.5, # Small, crisp screen pixel radius (stays sharp at all zoom levels)
                                "radiusUnit": "pixels", 
                                "opacity": 0.95, 
                                "filled": True, 
                                "outline": True, 
                                "thickness": 1.0, # Dark border provides contrast on top of the census shapes
                                "strokeColor": [15, 15, 15], 
                                "colorRange": color_scale_config
                            }
                        },
                        "visualChannels": {"colorField": {"name": "reliability", "type": "real"}, "colorScale": "quantize"}
                    },
                    {
                        "id": "segments", "type": "geojson",
                        "config": {
                            "dataId": "segments", "label": "Route Segments", "columns": {"geojson": "geometry"}, "isVisible": True,
                            "visConfig": {
                                "opacity": 0.9, 
                                "strokeOpacity": 0.9, 
                                "thickness": 1.4, 
                                "strokeColor": None, 
                                "colorRange": color_scale_config, 
                                "strokeColorRange": color_scale_config
                            }
                        },
                        "visualChannels": {"colorField": {"name": "avg_reliability", "type": "real"}, "colorScale": "quantize", "strokeColorField": {"name": "avg_reliability", "type": "real"}, "strokeColorScale": "quantize"}
                    },
                    {
                        "id": "eq_income", "type": "geojson",
                        "config": {
                            "dataId": "equity", "label": "Median Household Income ($)", "columns": {"geojson": "geometry"}, "isVisible": True,
                            "visConfig": {"opacity": 0.22, "strokeOpacity": 0.25, "thickness": 0.2, "strokeColor": [180, 180, 180], "filled": True, "enable3d": False, "colorRange": {"name": "Income_Blues", "type": "custom", "category": "Custom", "colors": ["#eff3ff","#c6dbef","#9ecae1","#6baed6","#3182bd","#08519c"]}}
                        },
                        "visualChannels": {"colorField": {"name": "median_income", "type": "real"}, "colorScale": "quantile"}
                    },
                    {
                        "id": "eq_lowincome", "type": "geojson",
                        "config": {
                            "dataId": "equity", "label": "Low-Income Households (%)", "columns": {"geojson": "geometry"}, "isVisible": False,
                            "visConfig": {"opacity": 0.22, "strokeOpacity": 0.25, "thickness": 0.2, "strokeColor": [180, 180, 180], "filled": True, "enable3d": False, "colorRange": {"name": "LowIncome_Purples", "type": "custom", "category": "Custom", "colors": ["#f2f0f7","#dadaeb","#bcbddc","#9e9ac8","#756bb1","#54278f"]}}
                        },
                        "visualChannels": {"colorField": {"name": "low_income_pct", "type": "real"}, "colorScale": "quantile"}
                    },
                    {
                        "id": "eq_transit", "type": "geojson",
                        "config": {
                            "dataId": "equity", "label": "Transit Commuters (%) — Transit Dependence", "columns": {"geojson": "geometry"}, "isVisible": False,
                            "visConfig": {"opacity": 0.22, "strokeOpacity": 0.25, "thickness": 0.2, "strokeColor": [180, 180, 180], "filled": True, "enable3d": False, "colorRange": {"name": "Transit_Pinks", "type": "custom", "category": "Custom", "colors": ["#fde0dd","#fcc5c0","#fa9fb5","#f768a1","#c51b8a","#7a0177"]}}
                        },
                        "visualChannels": {"colorField": {"name": "transit_commute_pct", "type": "real"}, "colorScale": "quantile"}
                    },
                    {
                        "id": "eq_vismin", "type": "geojson",
                        "config": {
                            "dataId": "equity", "label": "Visible Minority Population (%)", "columns": {"geojson": "geometry"}, "isVisible": False,
                            "visConfig": {"opacity": 0.22, "strokeOpacity": 0.25, "thickness": 0.2, "strokeColor": [180, 180, 180], "filled": True, "enable3d": False, "colorRange": {"name": "VisMin_Greys", "type": "custom", "category": "Custom", "colors": ["#f7f7f7","#d9d9d9","#bdbdbd","#969696","#636363","#252525"]}}
                        },
                        "visualChannels": {"colorField": {"name": "visible_minority_pct", "type": "real"}, "colorScale": "quantile"}
                    },
                    {
                        "id": "eq_immigrant", "type": "geojson",
                        "config": {
                            "dataId": "equity", "label": "Recent Immigrants — Last 5 Years (%)", "columns": {"geojson": "geometry"}, "isVisible": False,
                            "visConfig": {"opacity": 0.22, "strokeOpacity": 0.25, "thickness": 0.2, "strokeColor": [180, 180, 180], "filled": True, "enable3d": False, "colorRange": {"name": "Immigrant_Indigo", "type": "custom", "category": "Custom", "colors": ["#bfd3e6","#9ebcda","#8c96c6","#8c6bb1","#88419d","#810f7c"]}}
                        },
                        "visualChannels": {"colorField": {"name": "recent_immigrant_pct", "type": "real"}, "colorScale": "quantile"}
                    },
                    {
                        "id": "eq_seniors", "type": "geojson",
                        "config": {
                            "dataId": "equity", "label": "Seniors 65+ (%)", "columns": {"geojson": "geometry"}, "isVisible": False,
                            "visConfig": {"opacity": 0.22, "strokeOpacity": 0.25, "thickness": 0.2, "strokeColor": [180, 180, 180], "filled": True, "enable3d": False, "colorRange": {"name": "Senior_Browns", "type": "custom", "category": "Custom", "colors": ["#f6e8c3","#dfc27d","#bf812d","#8c510a","#543005","#331A00"]}}
                        },
                        "visualChannels": {"colorField": {"name": "senior_pct", "type": "real"}, "colorScale": "quantile"}
                    }
                ],
                "layerOrder": ["stops", "segments", "eq_income", "eq_lowincome", "eq_transit", "eq_vismin", "eq_immigrant", "eq_seniors"],
                "interactionConfig": {
                    "tooltip": {
                        "fieldsToShow": {
                            "equity": [
                                {"name": "area_name", "format": None},
                                {"name": "median_income", "format": None},
                                {"name": "low_income_pct", "format": None},
                                {"name": "transit_commute_pct", "format": None},
                                {"name": "visible_minority_pct", "format": None},
                                {"name": "recent_immigrant_pct", "format": None},
                                {"name": "senior_pct", "format": None}
                            ],
                            "segments": [{"name": "segment", "format": None}, {"name": "route_id", "format": None}, {"name": "avg_reliability", "format": ".1f"}],
                            "stops": [{"name": "stop_name", "format": None}, {"name": "route_id", "format": None}, {"name": "reliability", "format": ".1f"}]
                        },
                        "enabled": True
                    }
                }
            },
            "mapStyle": {"styleType": "muted_night"}
        }
    }
