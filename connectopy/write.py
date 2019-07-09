import h5py

def save_model(lm, model_path, parameters={'global': None, 'model': None}):

    """
    Save the fitted model.

    Parameters:
    - - - - -
    lm: object
        fitted linear model
    global_keys: dictionary
        global parameters for the model
        these include paths to x / y features, and subject ID
    model_params: dictionary
        parameters for specific model
        these include atlas type (Desikan or Destrieux), 
        dimensionality, and neighborhood size, etc.
    """

    model = h5py.File(model_path, mode='a')

    global_params = parameters['global']
    model_params = parameters['model']
    
    # add / update global parameters
    for (k, v) in global_params.items():
        model.attrs[k] = v
    
    # update number of models
    try:
        linear_models = model.create_group(name='models')
    except ValueError:
        linear_models = model['models']

    try:
        model.attrs['n_models'] += 1
    except KeyError:
        model.attrs['n_models'] = 1
    else:
        pass
    finally:
        n_models = model.attrs['n_models']
        model_id = 'm_' + str(n_models)

    # add current model
    current_model = linear_models.create_group(name=model_id)
    for (k, v) in model_params.items():
        current_model.attrs[k] = v

    current_model.create_dataset(name='r2', data=lm.r2_)
    current_model.create_dataset(name='coefficients', data=lm.coefficients_)

    model.close()

    