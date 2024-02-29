In the `KPJC6` main directory, please clone the public Montepython version like this:
    
    git clone git@github.com:brinckmann/montepython_public.git

or use another auth method in https://github.com/brinckmann/montepython_public)

To set up the path to class you need to do:
    
    cd montepython_public
    
    cp default.conf.template default.conf

then edit default.conf and set the path "root" to the directory that contains `class_public/` and change `path['cosmo'] accordingly.

Otherwise copy the default conf provided in this repo:

    cp photometric_fofR/MP_default.conf default.conf

The f(R) likelihood, plus needed data and snippets are not publicly available yet, so copy them by returning to the main directory and doing:

    cd ..
    cp -v photometric_fofR/codes/MGfit_Winther.py montepython_public/montepython/
    cp -v photometric_fofR/codes/scaledmeanlum_E2Sa.dat montepython_public/data/
    cp -vr photometric_fofR/codes/euclid_photometric_z_fofr montepython_public/montepython/likelihoods
