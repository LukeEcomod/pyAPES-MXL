#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 12:39:09 2018

@author: ajkieloaho
"""

ranges = {}

parameters = {
        'count': 1,
        'canopy': {
                'forestfloor': {
                        'bryophytes': {
                                'hylocomium': {
                                        'ground_coverage': (0.3),
                                        },
                                'sphagnum': {
                                        'ground_coverage': (0.3),
                                        },
                                'pleurozium': {
                                        'ground_coverage': (0.0),
                                        }
                                },
                        'litter': {
                                'ground_coverage': (0.2)
                                },
                        'baresoil': {
                                'ground_coverage': (0.2)
                                }
                        },
        }
    }

def iterate_parameters(parameters, default, count):
    """ Going through recursively senstivity nested parameter dictionary.

    Args:
        paramters (dict): nested dictionary
    """

    for key, value in parameters.items():
        if key == 'count':
            continue
        elif isinstance(value, dict):
            default[key] = iterate_parameters(value, default[key], count)

        else:
            if isinstance(value, tuple):
                default[key] = value[count]
            else:
                default[key] = value

    return default