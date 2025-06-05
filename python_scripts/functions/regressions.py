def regressions(ddf, features, target="fj_fs", reg="theilsen", zero_intercept=False, verbose=True):
    """
    Perform regression analysis using various methods.
    
    Args:
        ddf: Input DataFrame
        features: List of feature columns
        target: Target variable column
        reg: Regression method ('ols', 'ransac', 'theilsen', 'odr')
        zero_intercept: Force intercept through zero if True
        verbose: Print regression results if True
    
    Returns:
        Dictionary containing regression results
    """
    from sklearn import linear_model
    from sklearn.linear_model import LinearRegression, RANSACRegressor, HuberRegressor, TheilSenRegressor
    from scipy import odr
    from numpy import array, std, ones, sum, mean, ones_like

    # Validate regression method
    valid_methods = ['ols', 'ransac', 'theilsen', 'odr']
    if reg.lower() not in valid_methods:
        raise ValueError(f"Invalid regression method. Must be one of {valid_methods}")

    _df = ddf.copy()

    # Remove time and target from features
    if target in features:
        features.remove(target)

    # remove time from features
    if "time" in features:
        features.remove("time")

    # Define x and y data
    X = _df[features].values.reshape(-1, len(features))
    y = _df[target].values

    # Initialize predictions list and model
    model_predict = []
    model = None

    # Multi linear regression
    if reg.lower() == "ols":
        model = linear_model.LinearRegression(fit_intercept=not zero_intercept)
        model.fit(X, y)
        if verbose:
            print("R2:", model.score(X, y))
            if not zero_intercept:
                print("X0:", model.intercept_)
            print("Coef: ", model.coef_)
            for _f, _c in zip(features, model.coef_):
                print(f"{_f} : {_c}")
        
        # Make predictions
        for o, row in _df[features].iterrows():
            x_pred = array([row[feat] for feat in features]).reshape(-1, len(features))
            model_predict.append(model.predict(x_pred)[0])

    elif reg.lower() == "ransac":
        # For newer scikit-learn versions, use estimator instead of base_estimator
        try:
            model = RANSACRegressor(
                estimator=LinearRegression(fit_intercept=not zero_intercept),
                random_state=1
            ).fit(X, y)
        except TypeError:
            # Fallback for older versions
            model = RANSACRegressor(
                base_estimator=LinearRegression(fit_intercept=not zero_intercept),
                random_state=1
            ).fit(X, y)
            
        if verbose:
            print("R2:", model.score(X, y))
            if not zero_intercept:
                print("IC: ", model.estimator_.intercept_)
            print("Coef: ", model.estimator_.coef_)
            for _f, _c in zip(features, model.estimator_.coef_):
                print(f"{_f} : {_c}")
        
        # Make predictions
        for o, row in _df[features].iterrows():
            x_pred = array([row[feat] for feat in features]).reshape(-1, len(features))
            model_predict.append(model.predict(x_pred)[0])

    elif reg.lower() == "theilsen":
        model = TheilSenRegressor(fit_intercept=not zero_intercept).fit(X, y)
        if verbose:
            print("R2:", model.score(X, y))
            if not zero_intercept:
                print("X0:", model.intercept_)
            print("Coef: ", model.coef_)
            for _f, _c in zip(features, model.coef_):
                print(f"{_f} : {_c}")
        
        # Make predictions
        for o, row in _df[features].iterrows():
            x_pred = array([row[feat] for feat in features]).reshape(-1, len(features))
            model_predict.append(model.predict(x_pred)[0])

    elif reg.lower() == "odr":
        # Define ODR model function for single feature
        def f(B, x):
            if zero_intercept:
                return B[0] * x
            return B[0] * x + B[1]
        
        # Create ODR model
        linear = odr.Model(f)
        
        # Prepare data for ODR (ensure correct shapes)
        X_odr = X.reshape(-1)  # Flatten for single feature
        
        # Estimate data uncertainties if not provided
        sx = std(X_odr) * ones_like(X_odr)
        sy = std(y) * ones_like(y)
        
        # Create ODR data object
        data = odr.RealData(X_odr, y, sx=sx, sy=sy)
        
        # Set initial parameter guess
        if zero_intercept:
            beta0 = [1.0]
        else:
            beta0 = [1.0, 0.0]
        
        # Create ODR object and fit
        odr_obj = odr.ODR(data, linear, beta0=beta0)
        model = odr_obj.run()
        
        if verbose:
            print("R2:", 1 - (model.sum_square / sum((y - mean(y))**2)))
            print("Parameters:", model.beta)
            print("Parameter errors:", model.sd_beta)
            if not zero_intercept:
                print("Intercept:", model.beta[1])
            print(f"Slope: {model.beta[0]}")

        # Make predictions for ODR
        for o, row in _df[features].iterrows():
            x_pred = array([row[feat] for feat in features]).reshape(-1)[0]  # Get single value
            if zero_intercept:
                pred = model.beta[0] * x_pred
            else:
                pred = model.beta[0] * x_pred + model.beta[1]
            model_predict.append(pred)

    # Verify model was created
    if model is None:
        raise RuntimeError("Failed to create regression model")

    # Prepare output dictionary
    out = {
        'model': model,
        'r2': (1 - (model.sum_square / sum((y - mean(y))**2))) if reg.lower() == "odr" 
              else model.score(X, y),
        'tp': _df.time,
        'dp': model_predict
    }

    # Add slope and intercept based on regression method
    if reg.lower() == "ransac":
        out['slope'] = model.estimator_.coef_[0]
        out['inter'] = 0.0 if zero_intercept else model.estimator_.intercept_
    elif reg.lower() == "theilsen":
        out['slope'] = model.coef_[0]
        out['inter'] = 0.0 if zero_intercept else model.intercept_
    elif reg.lower() == "ols":
        out['slope'] = model.coef_[0]
        out['inter'] = 0.0 if zero_intercept else model.intercept_
    elif reg.lower() == "odr":
        out['slope'] = model.beta[0]
        out['inter'] = 0.0 if zero_intercept else model.beta[1]

    return out