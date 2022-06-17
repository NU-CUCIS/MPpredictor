from flask import Flask, render_template, request, Markup
from sklearn import linear_model, svm, ensemble, tree, neighbors
import pickle
from functions import *
import numpy as np

architecture = '1024x4D-512x3D-256x3D-128x3D-64x2-32x1-1'
activation = 'relu'
dropouts = [0.8, 0.9, 0.7, 0.8]
num_input = 86
num_input_pa = 145
num_input_fe1024 = 1024
properties = ["Properties", "Encut (eV)", "Kpoint Length Unit (Å)", "Kpoints Array Average (Å)", "Bandgap Optb88vdw (eV)", 
              "Formation Enthalpy (eV/atom)","Ehull (eV/atom)", "Magmom Oszicar (μB)", "Magmom Outcar (μB)", "Eps Average Root (x)", 
              "P Effective Masses (kg)", "N Effective Masses (kg)", "P Powerfact (W{}m{}K)".format('\u2081', '\u2081'), "N Powerfact (W{}m{}K)".format('\u2081', '\u2081'), 
              "P Seebeck (μV/K)", "N Seebeck (μV/K)", "Meps Average Root (x)", "Max Mode (cm{}{})".format('\u207B', '\u00B9'), "Min Mode (cm{}{})".format('\u207B', '\u00B9'), "Elastic Tensor C11 (GPa)", 
              "Elastic Tensor C12 (GPa)", "Elastic Tensor C13 (GPa)", "Elastic Tensor C22 (GPa)", "Elastic Tensor C33 (GPa)",
              "Elastic Tensor C44 (GPa)", "Elastic Tensor C55 (GPa)", "Elastic Tensor C66 (GPa)", "Bulk Modulus Kv (GPa)", 
              "Shear Modulus Gv (GPa)", "Bandgap MBJ (eV)", "Spillage (Å{}{})".format('\u207B', '\u00B9'), "Slme (%)", "Max Ir Mode (cm{}{})".format('\u207B', '\u00B9'), "Min Ir Mode (cm{}{})".format('\u207B', '\u00B9'),
              "Dfpt PM Dielectric Electronic (ε{}{})".format('\u2081', '\u2081'), "Dfpt PM Dielectric (ε{}{})".format('\u2081', '\u2081'), "Dfpt PM Dielectric Ioonic (ε{}{})".format('\u2081', '\u2081'), 
              "Dfpt PM Eij (cm{}{})".format('\u207B', '\u00B2'), "Dfpt PM Dij (cm{}{})".format('\u207B', '\u00B2'), "Exfoliation Energy (eV/atom)", "Expt Formation Energy (eV/atom)", "Expt Bandgap (eV)"]

app = Flask(__name__)

inputs = Input(shape=(num_input,), name='elemental_fractions')
outputs = define_model(inputs, architecture, dropouts=dropouts)
model = Model(inputs=inputs, outputs=outputs, name= 'ElemNet')

inputs_pa = Input(shape=(num_input_pa,), name='physical_attributes')
outputs_pa = define_model(inputs_pa, architecture, dropouts=dropouts)
model_pa = Model(inputs=inputs_pa, outputs=outputs_pa, name= 'ElemNet')

inputs_fe1024 = Input(shape=(num_input_fe1024,), name='input1024')
outputs_fe1024 = define_model(inputs_fe1024, architecture, dropouts=dropouts)
model_fe1024 = Model(inputs=inputs_fe1024, outputs=outputs_fe1024, name= 'ElemNet')

@app.route('/', methods = ['GET', 'POST'])
def index():
    return render_template('./index.html')

@app.route('/prediction.html', methods = ['GET', 'POST'])
def prediction():
    compounds = str(request.form.get('compounds', type=str).strip())
    try:
        compounds = toList(compounds)
        compound_ef = compound_to_ef(compounds)
        compound_pa = compound_to_pa(compounds)

        np_properties = np.asarray(properties)
        np_compounds = np.asarray(compounds)

        np_compounds = np.concatenate((np_properties[0].reshape(1,1), np_compounds.reshape(1,len(np_compounds))), axis=1)
        np_compounds = np_compounds.reshape(np_compounds.shape[1],) 

        if (request.form.get('all')!=None):

            encut = np.round(ml_model_prediction('model/model_ml_encut.sav', compound_ef), 1)
            np_encut = np.concatenate((np_properties[1].reshape(1,1), encut.reshape(1,encut.shape[0])), axis=1)        
            
            klu = np.round(ml_model_prediction('model/model_mlpa_kpoint_length_unit.sav', compound_pa), 4)
            np_klu = np.concatenate((np_properties[2].reshape(1,1), klu.reshape(1,klu.shape[0])), axis=1)
            
            kpaa = np.round(ml_model_prediction('model/model_mlpa_kpoints_array_average.sav', compound_pa), 4)        
            np_kpaa = np.concatenate((np_properties[3].reshape(1,1), kpaa.reshape(1,kpaa.shape[0])), axis=1)        
            
            optb = np.round(model_prediction(model, 'model/model_tlmod_optb88vdw_bandgap_fromstability_elemnet_tf2_1.h5', compound_ef), 4)
            np_optb = np.concatenate((np_properties[4].reshape(1,1), optb.reshape(1,optb.shape[0])), axis=1)
            
            deltae =  np.round(model_prediction(model, 'model/model_tl_deltae_fromdeltae_elemnet_tf2_1.h5', compound_ef), 4)
            np_deltae = np.concatenate((np_properties[5].reshape(1,1), deltae.reshape(1,deltae.shape[0])), axis=1)
            
            ehull =  np.round(model_prediction(model, 'model/model_tl_ehull_fromdeltae_elemnet_tf2_1.h5', compound_ef), 4)
            np_ehull = np.concatenate((np_properties[6].reshape(1,1), ehull.reshape(1,ehull.shape[0])), axis=1)
            
            magoszi =  np.round(model_prediction(model, 'model/model_tl_magoszi_frommagmom_elemnet_tf2_1.h5', compound_ef), 4)
            np_magoszi = np.concatenate((np_properties[7].reshape(1,1), magoszi.reshape(1,magoszi.shape[0])), axis=1)
        
            magout =  np.round(model_prediction(model, 'model/model_tlmod_magout_frommagmom_elemnet_tf2_1.h5', compound_ef), 4)
            np_magout = np.concatenate((np_properties[8].reshape(1,1), magout.reshape(1,magout.shape[0])), axis=1)
            
            eps =  np.round(model_prediction(model, 'model/model_tl_eps_fromdeltae_elemnet_tf2_1.h5', compound_ef), 4)
            np_eps = np.concatenate((np_properties[9].reshape(1,1), eps.reshape(1,eps.shape[0])), axis=1)
            
            pem300k = np.round(ml_model_prediction('model/model_mlpa_p_em300k.sav', compound_pa), 4)
            np_pem300k = np.concatenate((np_properties[10].reshape(1,1), pem300k.reshape(1,pem300k.shape[0])), axis=1)
            
            nem300k = np.round(ml_model_prediction('model/model_mlpa_n_em300k.sav', compound_pa), 4)
            np_nem300k = np.concatenate((np_properties[11].reshape(1,1), nem300k.reshape(1,nem300k.shape[0])), axis=1)
            
            ppf = np.round(model_prediction(model, 'model/model_tl_ppf_fromstability_elemnet_tf2_1.h5', compound_ef), 1)
            np_ppf = np.concatenate((np_properties[12].reshape(1,1), ppf.reshape(1,ppf.shape[0])), axis=1)
            
            npf = np.round(model_prediction(model, 'model/model_tl_npf_fromstability_elemnet_tf2_1.h5', compound_ef), 1)
            np_npf = np.concatenate((np_properties[13].reshape(1,1), npf.reshape(1,npf.shape[0])), axis=1)
            
            psb = np.round(ml_model_prediction('model/model_mlpa_p_Seebeck.sav', compound_pa), 1)
            np_psb = np.concatenate((np_properties[14].reshape(1,1), psb.reshape(1,psb.shape[0])), axis=1)
            
            nsb = np.round(ml_model_prediction('model/model_mlpa_n_Seebeck.sav', compound_pa), 1)
            np_nsb = np.concatenate((np_properties[15].reshape(1,1), nsb.reshape(1,nsb.shape[0])), axis=1)
            
            meps = np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_nature_wojarvis_new_training-false_tf2_1.h5', 'model/model_feattl_meps_fromdeltae_layer4_tf2_1.h5', compound_ef, 2), 4)
            np_meps = np.concatenate((np_properties[16].reshape(1,1), meps.reshape(1,meps.shape[0])), axis=1)
            
            maxm = np.round(ml_model_prediction('model/model_mlpa_max_mode.sav', compound_pa), 1)
            np_maxm = np.concatenate((np_properties[17].reshape(1,1), maxm.reshape(1,maxm.shape[0])), axis=1)        
            
            minm =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_naturestability_wojarvis_new_trf_tf2_1.h5', 'model/model_feattl_minm_fromstability_layer8_tf2_1.h5', compound_ef, 4), 2)
            np_minm = np.concatenate((np_properties[18].reshape(1,1), minm.reshape(1,minm.shape[0])), axis=1)
            
            etc11 = np.round(model_prediction(model, 'model/model_tl_etc11_fromstability_elemnet_tf2_1.h5', compound_ef), 2)         
            np_etc11 = np.concatenate((np_properties[19].reshape(1,1), etc11.reshape(1,etc11.shape[0])), axis=1)
            
            etc12 =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_natureenergypa_wojarvis_new_trf_tf2_1.h5', 'model/model_feattl_etc12_fromenergy_layer4_tf2_1.h5', compound_ef, 2), 2)
            np_etc12 = np.concatenate((np_properties[20].reshape(1,1), etc12.reshape(1,etc12.shape[0])), axis=1)
            
            etc13 = np.round(model_prediction(model, 'model/model_tl_etc13_fromstability_elemnet_tf2_1.h5', compound_ef), 2)        
            np_etc13 = np.concatenate((np_properties[21].reshape(1,1), etc13.reshape(1,etc13.shape[0])), axis=1)        
            
            etc22 =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_natureenergypa_wojarvis_new_trf_tf2_1.h5', 'model/model_feattl_etc22_fromenergy_layer4_tf2_1.h5', compound_ef, 2), 2)                
            np_etc22 = np.concatenate((np_properties[22].reshape(1,1), etc22.reshape(1,etc22.shape[0])), axis=1)
            
            etc33 =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_nature_wojarvis_new_training-false_tf2_1.h5', 'model/model_feattl_etc33_fromdeltae_layer6_tf2_1.h5', compound_ef, 3), 2)        
            np_etc33 = np.concatenate((np_properties[23].reshape(1,1), etc33.reshape(1,etc33.shape[0])), axis=1)
            
            etc44 =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_naturestability_wojarvis_new_trf_tf2_1.h5', 'model/model_feattl_etc44_fromstability_layer6_tf2_1.h5', compound_ef, 3), 2)                
            np_etc44 = np.concatenate((np_properties[24].reshape(1,1), etc44.reshape(1,etc44.shape[0])), axis=1)
            
            etc55 =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_naturestability_wojarvis_new_trf_tf2_1.h5', 'model/model_feattl_etc55_fromstability_layer6_tf2_1.h5', compound_ef, 3), 2)                
            np_etc55 = np.concatenate((np_properties[25].reshape(1,1), etc55.reshape(1,etc55.shape[0])), axis=1)
            
            etc66 =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_naturestability_wojarvis_new_trf_tf2_1.h5', 'model/model_feattl_etc66_fromstability_layer6_tf2_1.h5', compound_ef, 3), 2)                
            np_etc66 = np.concatenate((np_properties[26].reshape(1,1), etc66.reshape(1,etc66.shape[0])), axis=1)
            
            bulk =  np.round(model_prediction(model, 'model/model_tl_bulkmoduluskv_fromstability_elemnet_tf2_1.h5', compound_ef), 2)
            np_bulk = np.concatenate((np_properties[27].reshape(1,1), bulk.reshape(1,bulk.shape[0])), axis=1)
            
            shear =  np.round(model_prediction(model, 'model/model_tl_shearmodulusgv_fromstability_elemnet_tf2_1.h5', compound_ef), 2)               
            np_shear = np.concatenate((np_properties[28].reshape(1,1), shear.reshape(1,shear.shape[0])), axis=1)
            
            mbj =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_naturebandgap_wojarvis_new_trf_tf2_1.h5', 'model/model_feattl_mbj_bandgap_frombandgap_layer2_tf2_1.h5', compound_ef, 1), 4)                
            np_mbj = np.concatenate((np_properties[29].reshape(1,1), mbj.reshape(1,mbj.shape[0])), axis=1)
            
            spillage =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_nature_wojarvis_new_training-false_tf2_1.h5', 'model/model_feattl_spillage_fromdeltae_layer4_tf2_1.h5', compound_ef, 2), 4)                
            np_spillage = np.concatenate((np_properties[30].reshape(1,1), spillage.reshape(1,spillage.shape[0])), axis=1)
            
            slme =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_naturemagmompa_wojarvis_range_new_trf_tf2_1.h5', 'model/model_feattl_slme_frommagmom_layer6_tf2_1.h5', compound_ef, 3), 4)                
            np_slme = np.concatenate((np_properties[31].reshape(1,1), slme.reshape(1,slme.shape[0])), axis=1)
            
            maxirm =  np.round(model_prediction(model_pa, 'model/model_scpa_maxirm_elemnet_tf2_1.h5', compound_pa), 1)
            np_maxirm = np.concatenate((np_properties[32].reshape(1,1), maxirm.reshape(1,maxirm.shape[0])), axis=1)
            
            minirm = np.round(ml_model_prediction('model/model_mlpa_min_ir_mode.sav', compound_pa), 1)
            np_minirm = np.concatenate((np_properties[33].reshape(1,1), minirm.reshape(1,minirm.shape[0])), axis=1)        
            
            pmdiel =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_naturestability_wojarvis_new_trf_tf2_1.h5', 'model/model_feattl_pmdiel_fromstability_layer2_tf2_1.h5', compound_ef, 1), 4)                
            np_pmdiel = np.concatenate((np_properties[34].reshape(1,1), pmdiel.reshape(1,pmdiel.shape[0])), axis=1)
            
            pmdielectric =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_naturestability_wojarvis_new_trf_tf2_1.h5', 'model/model_feattl_pmdi_fromstability_layer6_tf2_1.h5', compound_ef, 3), 4)                
            np_pmdielectric = np.concatenate((np_properties[35].reshape(1,1), pmdielectric.reshape(1,pmdielectric.shape[0])), axis=1)
            
            pmdiio =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_natureenergypa_wojarvis_new_trf_tf2_1.h5', 'model/model_feattl_pmdiio_fromenergy_layer2_tf2_1.h5', compound_ef, 1), 4)                
            np_pmdiio = np.concatenate((np_properties[36].reshape(1,1), pmdiio.reshape(1,pmdiio.shape[0])), axis=1)
            
            pmeij =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_naturebandgap_wojarvis_new_trf_tf2_1.h5', 'model/model_feattl_pmeij_frombandgap_layer6_tf2_1.h5', compound_ef, 3), 4)                
            np_pmeij = np.concatenate((np_properties[37].reshape(1,1), pmeij.reshape(1,pmeij.shape[0])), axis=1)        
            
            pmdij =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_natureenergypa_wojarvis_new_trf_tf2_1.h5', 'model/model_feattl_pmdij_fromenergy_layer6_tf2_1.h5', compound_ef, 3), 2)                
            np_pmdij = np.concatenate((np_properties[38].reshape(1,1), pmdij.reshape(1,pmdij.shape[0])), axis=1)
            
            exfoli =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_nature_wojarvis_new_training-false_tf2_1.h5', 'model/model_feattl_exfoli_fromdeltae_layer8_tf2_1.h5', compound_ef, 4), 2)                
            np_exfoli = np.concatenate((np_properties[39].reshape(1,1), exfoli.reshape(1,exfoli.shape[0])), axis=1)
            
            exptdeltae =  np.round(model_prediction(model, 'model/model_tl_expt_deltae_fromdeltae_elemnet_tf2_1.h5', compound_ef), 4)
            np_exptdeltae = np.concatenate((np_properties[40].reshape(1,1), exptdeltae.reshape(1,exptdeltae.shape[0])), axis=1)
            
            exptbandgap =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_nature_wojarvis_new_training-false_tf2_1.h5', 'model/model_feattl_expt_bandgap_fromdeltae_layer4_tf2_1.h5', compound_ef, 2), 4)        
            np_exptbandgap = np.concatenate((np_properties[41].reshape(1,1), exptbandgap.reshape(1,exptbandgap.shape[0])), axis=1)
            

            results = np.concatenate((np_encut, np_klu, np_kpaa, np_optb, np_deltae, np_ehull, np_magoszi, np_magout, 
                                    np_eps, np_pem300k, np_nem300k, np_ppf, np_npf, np_psb, np_nsb, np_meps, np_maxm, 
                                    np_minm, np_etc11, np_etc12, np_etc13, np_etc22, np_etc33, np_etc44, np_etc55, np_etc66, 
                                    np_bulk, np_shear, np_mbj, np_spillage, np_slme, np_maxirm, np_minirm, np_pmdiel, 
                                    np_pmdielectric, np_pmdiio, np_pmeij, np_pmdij, np_exfoli, np_exptdeltae, np_exptbandgap), axis=0)
            print("hello")
        else:   
            select_property = []
            select_result = []
            if(request.form.get('encut2')!=None):
                encut = np.round(ml_model_prediction('model/model_ml_encut.sav', compound_ef), 1)
                np_encut = np.concatenate((np_properties[1].reshape(1,1), encut.reshape(1,encut.shape[0])), axis=1)        
                select_property.append(np_properties[1])
                select_result.append(np_encut)

            if(request.form.get('klu2')!=None):
                klu = np.round(ml_model_prediction('model/model_mlpa_kpoint_length_unit.sav', compound_pa), 4)
                np_klu = np.concatenate((np_properties[2].reshape(1,1), klu.reshape(1,klu.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_klu)
                
            if(request.form.get('kpaa2')!=None):
                kpaa = np.round(ml_model_prediction('model/model_mlpa_kpoints_array_average.sav', compound_pa), 4)        
                np_kpaa = np.concatenate((np_properties[3].reshape(1,1), kpaa.reshape(1,kpaa.shape[0])), axis=1)          
                select_property.append(np_properties[1])
                select_result.append(np_kpaa)
                
            if(request.form.get('bgoptb2')!=None):
                optb = np.round(model_prediction(model, 'model/model_tlmod_optb88vdw_bandgap_fromstability_elemnet_tf2_1.h5', compound_ef), 4)
                np_optb = np.concatenate((np_properties[4].reshape(1,1), optb.reshape(1,optb.shape[0])), axis=1) 
                select_property.append(np_properties[1])
                select_result.append(np_optb)
                
            if(request.form.get('deltae2')!=None):
                deltae =  np.round(model_prediction(model, 'model/model_tl_deltae_fromdeltae_elemnet_tf2_1.h5', compound_ef), 4)
                np_deltae = np.concatenate((np_properties[5].reshape(1,1), deltae.reshape(1,deltae.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_deltae)
                
            if(request.form.get('ehull2')!=None):
                ehull =  np.round(model_prediction(model, 'model/model_tl_ehull_fromdeltae_elemnet_tf2_1.h5', compound_ef), 4)
                np_ehull = np.concatenate((np_properties[6].reshape(1,1), ehull.reshape(1,ehull.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_ehull)
                
            if(request.form.get('magoszi2')!=None):
                magoszi =  np.round(model_prediction(model, 'model/model_tl_magoszi_frommagmom_elemnet_tf2_1.h5', compound_ef), 4)
                np_magoszi = np.concatenate((np_properties[7].reshape(1,1), magoszi.reshape(1,magoszi.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_magoszi)
                
            if(request.form.get('magout2')!=None):
                magout =  np.round(model_prediction(model, 'model/model_tlmod_magout_frommagmom_elemnet_tf2_1.h5', compound_ef), 4)
                np_magout = np.concatenate((np_properties[8].reshape(1,1), magout.reshape(1,magout.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_magout)
                
            if(request.form.get('eps2')!=None):
                eps =  np.round(model_prediction(model, 'model/model_tl_eps_fromdeltae_elemnet_tf2_1.h5', compound_ef), 4)
                np_eps = np.concatenate((np_properties[9].reshape(1,1), eps.reshape(1,eps.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_eps)
                
            if(request.form.get('pem300k2')!=None):
                pem300k = np.round(ml_model_prediction('model/model_mlpa_p_em300k.sav', compound_pa), 4)
                np_pem300k = np.concatenate((np_properties[10].reshape(1,1), pem300k.reshape(1,pem300k.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_pem300k)
                
            if(request.form.get('nem300k2')!=None):
                nem300k = np.round(ml_model_prediction('model/model_mlpa_n_em300k.sav', compound_pa), 4)
                np_nem300k = np.concatenate((np_properties[11].reshape(1,1), nem300k.reshape(1,nem300k.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_nem300k)
                
            if(request.form.get('ppf2')!=None):
                ppf = np.round(model_prediction(model, 'model/model_tl_ppf_fromstability_elemnet_tf2_1.h5', compound_ef), 1)
                np_ppf = np.concatenate((np_properties[12].reshape(1,1), ppf.reshape(1,ppf.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_ppf)
                
            if(request.form.get('npf2')!=None):
                npf = np.round(model_prediction(model, 'model/model_tl_npf_fromstability_elemnet_tf2_1.h5', compound_ef), 1)
                np_npf = np.concatenate((np_properties[13].reshape(1,1), npf.reshape(1,npf.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_npf)
                
            if(request.form.get('psb2')!=None):
                psb = np.round(ml_model_prediction('model/model_mlpa_p_Seebeck.sav', compound_pa), 1)
                np_psb = np.concatenate((np_properties[14].reshape(1,1), psb.reshape(1,psb.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_psb)
                
            if(request.form.get('nsb2')!=None):
                nsb = np.round(ml_model_prediction('model/model_mlpa_n_Seebeck.sav', compound_pa), 1)
                np_nsb = np.concatenate((np_properties[15].reshape(1,1), nsb.reshape(1,nsb.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_nsb)
                
            if(request.form.get('meps2')!=None):
                meps =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_nature_wojarvis_new_training-false_tf2_1.h5', 'model/model_feattl_meps_fromdeltae_layer4_tf2_1.h5', compound_ef, 2), 4)
                np_meps = np.concatenate((np_properties[16].reshape(1,1), meps.reshape(1,meps.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_meps)
                
            if(request.form.get('maxm2')!=None):
                maxm = np.round(ml_model_prediction('model/model_mlpa_max_mode.sav', compound_pa), 1)
                np_maxm = np.concatenate((np_properties[17].reshape(1,1), maxm.reshape(1,maxm.shape[0])), axis=1)        
                select_property.append(np_properties[1])
                select_result.append(np_maxm)
                
            if(request.form.get('minm2')!=None):
                minm =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_naturestability_wojarvis_new_trf_tf2_1.h5', 'model/model_feattl_minm_fromstability_layer8_tf2_1.h5', compound_ef, 4), 4)
                np_minm = np.concatenate((np_properties[18].reshape(1,1), minm.reshape(1,minm.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_minm)
                
            if(request.form.get('etc112')!=None):
                etc11 = np.round(model_prediction(model, 'model/model_tl_etc11_fromstability_elemnet_tf2_1.h5', compound_ef), 2)        
                np_etc11 = np.concatenate((np_properties[19].reshape(1,1), etc11.reshape(1,etc11.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_etc11)
                
            if(request.form.get('etc122')!=None):
                etc12 =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_natureenergypa_wojarvis_new_trf_tf2_1.h5', 'model/model_feattl_etc12_fromenergy_layer4_tf2_1.h5', compound_ef, 2), 2)
                np_etc12 = np.concatenate((np_properties[20].reshape(1,1), etc12.reshape(1,etc12.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_etc12)
                
            if(request.form.get('etc132')!=None):
                etc13 = np.round(model_prediction(model, 'model/model_tl_etc13_fromstability_elemnet_tf2_1.h5', compound_ef), 2)        
                np_etc13 = np.concatenate((np_properties[21].reshape(1,1), etc13.reshape(1,etc13.shape[0])), axis=1)        
                select_property.append(np_properties[1])
                select_result.append(np_etc13)
                
            if(request.form.get('etc222')!=None):
                etc22 =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_natureenergypa_wojarvis_new_trf_tf2_1.h5', 'model/model_feattl_etc22_fromenergy_layer4_tf2_1.h5', compound_ef, 2), 2)                
                np_etc22 = np.concatenate((np_properties[22].reshape(1,1), etc22.reshape(1,etc22.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_etc22)
                
            if(request.form.get('etc332')!=None):
                etc33 =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_nature_wojarvis_new_training-false_tf2_1.h5', 'model/model_feattl_etc33_fromdeltae_layer6_tf2_1.h5', compound_ef, 3), 2)        
                np_etc33 = np.concatenate((np_properties[23].reshape(1,1), etc33.reshape(1,etc33.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_etc33)
                
            if(request.form.get('etc442')!=None):
                etc44 =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_naturestability_wojarvis_new_trf_tf2_1.h5', 'model/model_feattl_etc44_fromstability_layer6_tf2_1.h5', compound_ef, 3), 2)                
                np_etc44 = np.concatenate((np_properties[24].reshape(1,1), etc44.reshape(1,etc44.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_etc44)
                
            if(request.form.get('etc552')!=None):
                etc55 =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_naturestability_wojarvis_new_trf_tf2_1.h5', 'model/model_feattl_etc55_fromstability_layer6_tf2_1.h5', compound_ef, 3), 2)                
                np_etc55 = np.concatenate((np_properties[25].reshape(1,1), etc55.reshape(1,etc55.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_etc55)
                
            if(request.form.get('etc662')!=None):
                etc66 =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_naturestability_wojarvis_new_trf_tf2_1.h5', 'model/model_feattl_etc66_fromstability_layer6_tf2_1.h5', compound_ef, 3), 2)                
                np_etc66 = np.concatenate((np_properties[26].reshape(1,1), etc66.reshape(1,etc66.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_etc66)
                
            if(request.form.get('bulk2')!=None):
                bulk =  np.round(model_prediction(model, 'model/model_tl_bulkmoduluskv_fromstability_elemnet_tf2_1.h5', compound_ef), 2)
                np_bulk = np.concatenate((np_properties[27].reshape(1,1), bulk.reshape(1,bulk.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_bulk)
                
            if(request.form.get('shear2')!=None):
                shear =  np.round(model_prediction(model, 'model/model_tl_shearmodulusgv_fromstability_elemnet_tf2_1.h5', compound_ef), 2)                
                np_shear = np.concatenate((np_properties[28].reshape(1,1), shear.reshape(1,shear.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_shear)
                
            if(request.form.get('bgmbj2')!=None):
                mbj =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_naturebandgap_wojarvis_new_trf_tf2_1.h5', 'model/model_feattl_mbj_bandgap_frombandgap_layer2_tf2_1.h5', compound_ef, 1), 4)                
                np_mbj = np.concatenate((np_properties[29].reshape(1,1), mbj.reshape(1,mbj.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_mbj)
                
            if(request.form.get('spillage2')!=None):
                spillage =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_nature_wojarvis_new_training-false_tf2_1.h5', 'model/model_feattl_spillage_fromdeltae_layer4_tf2_1.h5', compound_ef, 2), 4)                
                np_spillage = np.concatenate((np_properties[30].reshape(1,1), spillage.reshape(1,spillage.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_spillage)
                
            if(request.form.get('slme2')!=None):
                slme =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_naturemagmompa_wojarvis_range_new_trf_tf2_1.h5', 'model/model_feattl_slme_frommagmom_layer6_tf2_1.h5', compound_ef, 3), 4)                
                np_slme = np.concatenate((np_properties[31].reshape(1,1), slme.reshape(1,slme.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_slme)
                
            if(request.form.get('maxirm2')!=None):
                maxirm =  np.round(model_prediction(model_pa, 'model/model_scpa_maxirm_elemnet_tf2_1.h5', compound_pa), 1)
                np_maxirm = np.concatenate((np_properties[32].reshape(1,1), maxirm.reshape(1,maxirm.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_maxirm)
                
            if(request.form.get('minirm2')!=None):
                minirm = np.round(ml_model_prediction('model/model_mlpa_min_ir_mode.sav', compound_pa), 1)
                np_minirm = np.concatenate((np_properties[33].reshape(1,1), minirm.reshape(1,minirm.shape[0])), axis=1)        
                select_property.append(np_properties[1])
                select_result.append(np_minirm)
                
            if(request.form.get('pmdiel2')!=None):
                pmdiel =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_naturestability_wojarvis_new_trf_tf2_1.h5', 'model/model_feattl_pmdiel_fromstability_layer2_tf2_1.h5', compound_ef, 1), 4)                
                np_pmdiel = np.concatenate((np_properties[34].reshape(1,1), pmdiel.reshape(1,pmdiel.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_pmdiel)
                
            if(request.form.get('pmdi2')!=None):
                pmdielectric =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_naturestability_wojarvis_new_trf_tf2_1.h5', 'model/model_feattl_pmdi_fromstability_layer6_tf2_1.h5', compound_ef, 3), 4)                
                np_pmdielectric = np.concatenate((np_properties[35].reshape(1,1), pmdielectric.reshape(1,pmdielectric.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_pmdielectric)
                
            if(request.form.get('pmdiio2')!=None):
                pmdiio =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_natureenergypa_wojarvis_new_trf_tf2_1.h5', 'model/model_feattl_pmdiio_fromenergy_layer2_tf2_1.h5', compound_ef, 1), 4)                
                np_pmdiio = np.concatenate((np_properties[36].reshape(1,1), pmdiio.reshape(1,pmdiio.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_pmdiio)
                
            if(request.form.get('pmeij2')!=None):
                pmeij =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_naturebandgap_wojarvis_new_trf_tf2_1.h5', 'model/model_feattl_pmeij_frombandgap_layer6_tf2_1.h5', compound_ef, 3), 4)                
                np_pmeij = np.concatenate((np_properties[37].reshape(1,1), pmeij.reshape(1,pmeij.shape[0])), axis=1)        
                select_property.append(np_properties[1])
                select_result.append(np_pmeij)
                
            if(request.form.get('pmdij2')!=None):
                pmdij =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_natureenergypa_wojarvis_new_trf_tf2_1.h5', 'model/model_feattl_pmdij_fromenergy_layer6_tf2_1.h5', compound_ef, 3), 2)                
                np_pmdij = np.concatenate((np_properties[38].reshape(1,1), pmdij.reshape(1,pmdij.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_pmdij)
                
            if(request.form.get('exfoli2')!=None):        
                exfoli =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_nature_wojarvis_new_training-false_tf2_1.h5', 'model/model_feattl_exfoli_fromdeltae_layer8_tf2_1.h5', compound_ef, 4), 2)                
                np_exfoli = np.concatenate((np_properties[39].reshape(1,1), exfoli.reshape(1,exfoli.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_exfoli)

            if(request.form.get('exptdeltae2')!=None):        
                exptdeltae =  np.round(model_prediction(model, 'model/model_tl_expt_deltae_fromdeltae_elemnet_tf2_1.h5', compound_ef), 4)
                np_exptdeltae = np.concatenate((np_properties[40].reshape(1,1), exptdeltae.reshape(1,exptdeltae.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_exptdeltae)

            if(request.form.get('exptbandgap2')!=None):        
                exptbandgap =  np.round(model_prediction_fe(model, model_fe1024, 'model/model_elemnet_nature_wojarvis_new_training-false_tf2_1.h5', 'model/model_feattl_expt_bandgap_fromdeltae_layer4_tf2_1.h5', compound_ef, 2), 4)        
                np_exptbandgap = np.concatenate((np_properties[41].reshape(1,1), exptbandgap.reshape(1,exptbandgap.shape[0])), axis=1)
                select_property.append(np_properties[1])
                select_result.append(np_exptbandgap)    

            results = np.asarray(select_result).reshape(len(select_property), len(np_compounds))


    except:
        print("Invalid") 
        np_compounds = np.asarray("Invalid").reshape(1,)
        results = np.asarray("Invalid").reshape(1,1)


    return render_template('./prediction.html', compounds = np_compounds, results = results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5040)       