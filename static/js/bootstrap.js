function mouseOver() {
    document.getElementById("subject").style.display = 'block';
  }
  
function mouseOut() {
    document.getElementById("subject").style.display = 'none';
  }

function mouseOvertest() {
    document.getElementById("all").style.border = '1px solid #ccc';
  }
  
function mouseOuttest() {
    document.getElementById("all").style.border = 'none';
  }  

function mouseOverElectronicStructure() {
    document.getElementById("klu").style.border = '1px solid #000';
    document.getElementById("kpaa").style.border = '1px solid #000';
  }
  
function mouseOutElectronicStructure() {
    document.getElementById("klu").style.border = '1px solid #fff';
    document.getElementById("kpaa").style.border = '1px solid #fff';
  }  

function clickElectronicStructure() {
    document.getElementById("klu2").checked = !document.getElementById("klu2").checked;
    document.getElementById("kpaa2").checked = !document.getElementById("kpaa2").checked;
  }  

function mouseOverConductivity() {
    document.getElementById("bgoptb").style.border = '1px solid #000';
    document.getElementById("pem300k").style.border = '1px solid #000';
    document.getElementById("nem300k").style.border = '1px solid #000';
    document.getElementById("bgmbj").style.border = '1px solid #000';
  }
  
function mouseOutConductivity() {
    document.getElementById("bgoptb").style.border = '1px solid #fff';
    document.getElementById("pem300k").style.border = '1px solid #fff';
    document.getElementById("nem300k").style.border = '1px solid #fff';
    document.getElementById("bgmbj").style.border = '1px solid #fff';
  }  

function clickConductivity() {
    document.getElementById("bgoptb2").checked = !document.getElementById("bgoptb2").checked;
    document.getElementById("pem300k2").checked = !document.getElementById("pem300k2").checked;
    document.getElementById("nem300k2").checked = !document.getElementById("nem300k2").checked;
    document.getElementById("bgmbj2").checked = !document.getElementById("bgmbj2").checked;
  }    

function mouseOverStability() {
    document.getElementById("deltae").style.border = '1px solid #000';
    document.getElementById("ehull").style.border = '1px solid #000';
  }
  
function mouseOutStability() {
    document.getElementById("deltae").style.border = '1px solid #fff';
    document.getElementById("ehull").style.border = '1px solid #fff';
  }  

function clickStability() {
    document.getElementById("deltae2").checked = !document.getElementById("deltae2").checked;
    document.getElementById("ehull2").checked = !document.getElementById("ehull2").checked;
  }    

function mouseOverMagnetism() {
    document.getElementById("magoszi").style.border = '1px solid #000';
    document.getElementById("magout").style.border = '1px solid #000';
  }
  
function mouseOutMagnetism() {
    document.getElementById("magoszi").style.border = '1px solid #fff';
    document.getElementById("magout").style.border = '1px solid #fff';
  }  

function clickMagnetism() {
    document.getElementById("magoszi2").checked = !document.getElementById("magoszi2").checked;
    document.getElementById("magout2").checked = !document.getElementById("magout2").checked;
  }  
  
function mouseOverPhotoOptics() {
    document.getElementById("eps").style.border = '1px solid #000';
    document.getElementById("meps").style.border = '1px solid #000';
  }
  
function mouseOutPhotoOptics() {
    document.getElementById("eps").style.border = '1px solid #fff';
    document.getElementById("meps").style.border = '1px solid #fff';
  }  

function clickPhotoOptics() {
    document.getElementById("eps2").checked = !document.getElementById("eps2").checked;
    document.getElementById("meps2").checked = !document.getElementById("meps2").checked;
  }   
  
  function mouseOverElasticity() {
    document.getElementById("etc11").style.border = '1px solid #000';
    document.getElementById("etc12").style.border = '1px solid #000';
    document.getElementById("etc13").style.border = '1px solid #000';
    document.getElementById("etc22").style.border = '1px solid #000';
    document.getElementById("etc33").style.border = '1px solid #000';
    document.getElementById("etc44").style.border = '1px solid #000';
    document.getElementById("etc55").style.border = '1px solid #000';
    document.getElementById("etc66").style.border = '1px solid #000';
    document.getElementById("bulk").style.border = '1px solid #000';
    document.getElementById("shear").style.border = '1px solid #000';
  }
  
function mouseOutElasticity() {
    document.getElementById("etc11").style.border = '1px solid #fff';
    document.getElementById("etc12").style.border = '1px solid #fff';
    document.getElementById("etc13").style.border = '1px solid #fff';
    document.getElementById("etc22").style.border = '1px solid #fff';
    document.getElementById("etc33").style.border = '1px solid #fff';
    document.getElementById("etc44").style.border = '1px solid #fff';
    document.getElementById("etc55").style.border = '1px solid #fff';
    document.getElementById("etc66").style.border = '1px solid #fff';
    document.getElementById("bulk").style.border = '1px solid #fff';
    document.getElementById("shear").style.border = '1px solid #fff';
  }  

function clickElasticity() {
    document.getElementById("etc112").checked = !document.getElementById("etc112").checked;
    document.getElementById("etc122").checked = !document.getElementById("etc122").checked;
    document.getElementById("etc132").checked = !document.getElementById("etc132").checked;
    document.getElementById("etc222").checked = !document.getElementById("etc222").checked;
    document.getElementById("etc332").checked = !document.getElementById("etc332").checked;
    document.getElementById("etc442").checked = !document.getElementById("etc442").checked;
    document.getElementById("etc552").checked = !document.getElementById("etc552").checked;
    document.getElementById("etc662").checked = !document.getElementById("etc662").checked;
    document.getElementById("bulk2").checked = !document.getElementById("bulk2").checked;
    document.getElementById("shear2").checked = !document.getElementById("shear2").checked;
  } 
  
function mouseOverElectricPotential() {
    document.getElementById("pmdiel").style.border = '1px solid #000';
    document.getElementById("pmdi").style.border = '1px solid #000';
    document.getElementById("pmdiio").style.border = '1px solid #000';
    document.getElementById("pmeij").style.border = '1px solid #000';
    document.getElementById("pmdij").style.border = '1px solid #000';
  }
  
function mouseOutElectricPotential() {
    document.getElementById("pmdiel").style.border = '1px solid #fff';
    document.getElementById("pmdi").style.border = '1px solid #fff';
    document.getElementById("pmdiio").style.border = '1px solid #fff';
    document.getElementById("pmeij").style.border = '1px solid #fff';
    document.getElementById("pmdij").style.border = '1px solid #fff';
  }  

function clickElectricPotential() {
    document.getElementById("pmdiel2").checked = !document.getElementById("pmdiel2").checked;
    document.getElementById("pmdi2").checked = !document.getElementById("pmdi2").checked;
    document.getElementById("pmdiio2").checked = !document.getElementById("pmdiio2").checked;
    document.getElementById("pmeij2").checked = !document.getElementById("pmeij2").checked;
    document.getElementById("pmdij2").checked = !document.getElementById("pmdij2").checked;
  }    

function delete_Button(){document.getElementById("compounds").value = document.getElementById("compounds").value.slice(0, -1);}
function clear_Button(){document.getElementById("compounds").value = '';}
function space_Button(){document.getElementById("compounds").value += ' ';}
function num1_Button(){document.getElementById("compounds").value += '1';}
function num2_Button(){document.getElementById("compounds").value += '2';}
function num3_Button(){document.getElementById("compounds").value += '3';}
function num4_Button(){document.getElementById("compounds").value += '4';}
function num5_Button(){document.getElementById("compounds").value += '5';}
function num6_Button(){document.getElementById("compounds").value += '6';}
function num7_Button(){document.getElementById("compounds").value += '7';}
function num8_Button(){document.getElementById("compounds").value += '8';}
function num9_Button(){document.getElementById("compounds").value += '9';}
function num0_Button(){document.getElementById("compounds").value += '0';}
function H_Button(){document.getElementById("compounds").value += 'H';}
function Li_Button(){document.getElementById("compounds").value += 'Li';}
function Be_Button(){document.getElementById("compounds").value += 'Be';}
function B_Button(){document.getElementById("compounds").value += 'B';}
function C_Button(){document.getElementById("compounds").value += 'C';}
function N_Button(){document.getElementById("compounds").value += 'N';}
function O_Button(){document.getElementById("compounds").value += 'O';}
function F_Button(){document.getElementById("compounds").value += 'F';}
function Na_Button(){document.getElementById("compounds").value += 'Na';}
function Mg_Button(){document.getElementById("compounds").value += 'Mg';}
function Al_Button(){document.getElementById("compounds").value += 'Al';}
function Si_Button(){document.getElementById("compounds").value += 'Si';}
function P_Button(){document.getElementById("compounds").value += 'P';}
function S_Button(){document.getElementById("compounds").value += 'S';}
function Cl_Button(){document.getElementById("compounds").value += 'Cl';}
function K_Button(){document.getElementById("compounds").value += 'K';}
function Ca_Button(){document.getElementById("compounds").value += 'Ca';}
function Sc_Button(){document.getElementById("compounds").value += 'Sc';}
function Ti_Button(){document.getElementById("compounds").value += 'Ti';}
function V_Button(){document.getElementById("compounds").value += 'V';}
function Cr_Button(){document.getElementById("compounds").value += 'Cr';}
function Mn_Button(){document.getElementById("compounds").value += 'Mn';}
function Fe_Button(){document.getElementById("compounds").value += 'Fe';}
function Co_Button(){document.getElementById("compounds").value += 'Co';}
function Ni_Button(){document.getElementById("compounds").value += 'Ni';}
function Cu_Button(){document.getElementById("compounds").value += 'Cu';}
function Zn_Button(){document.getElementById("compounds").value += 'Zn';}
function Ga_Button(){document.getElementById("compounds").value += 'Ga';}
function Ge_Button(){document.getElementById("compounds").value += 'Ge';}
function As_Button(){document.getElementById("compounds").value += 'As';}
function Se_Button(){document.getElementById("compounds").value += 'Se';}
function Br_Button(){document.getElementById("compounds").value += 'Br';}
function Kr_Button(){document.getElementById("compounds").value += 'Kr';}
function Rb_Button(){document.getElementById("compounds").value += 'Rb';}
function Sr_Button(){document.getElementById("compounds").value += 'Sr';}
function Y_Button(){document.getElementById("compounds").value += 'Y';}
function Zr_Button(){document.getElementById("compounds").value += 'Zr';}
function Nb_Button(){document.getElementById("compounds").value += 'Nb';}
function Mo_Button(){document.getElementById("compounds").value += 'Mo';}
function Tc_Button(){document.getElementById("compounds").value += 'Tc';}
function Ru_Button(){document.getElementById("compounds").value += 'Ru';}
function Rh_Button(){document.getElementById("compounds").value += 'Rh';}
function Pd_Button(){document.getElementById("compounds").value += 'Pd';}
function Ag_Button(){document.getElementById("compounds").value += 'Ag';}
function Cd_Button(){document.getElementById("compounds").value += 'Cd';}
function In_Button(){document.getElementById("compounds").value += 'In';}
function Sn_Button(){document.getElementById("compounds").value += 'Sn';}
function Sb_Button(){document.getElementById("compounds").value += 'Sb';}
function Te_Button(){document.getElementById("compounds").value += 'Te';}
function I_Button(){document.getElementById("compounds").value += 'I';}
function Xe_Button(){document.getElementById("compounds").value += 'Xe';}
function Cs_Button(){document.getElementById("compounds").value += 'Cs';}
function Ba_Button(){document.getElementById("compounds").value += 'Ba';}
function La_Button(){document.getElementById("compounds").value += 'La';}
function Hf_Button(){document.getElementById("compounds").value += 'Hf';}
function Ta_Button(){document.getElementById("compounds").value += 'Ta';}
function W_Button(){document.getElementById("compounds").value += 'W';}
function Re_Button(){document.getElementById("compounds").value += 'Re';}
function Os_Button(){document.getElementById("compounds").value += 'Os';}
function Ir_Button(){document.getElementById("compounds").value += 'Ir';}
function Pt_Button(){document.getElementById("compounds").value += 'Pt';}
function Au_Button(){document.getElementById("compounds").value += 'Au';}
function Hg_Button(){document.getElementById("compounds").value += 'Hg';}
function Tl_Button(){document.getElementById("compounds").value += 'Tl';}
function Pb_Button(){document.getElementById("compounds").value += 'Pb';}
function Bi_Button(){document.getElementById("compounds").value += 'Bi';}
function Ac_Button(){document.getElementById("compounds").value += 'Ac';}
function Ce_Button(){document.getElementById("compounds").value += 'Ce';}
function Pr_Button(){document.getElementById("compounds").value += 'Pr';}
function Nd_Button(){document.getElementById("compounds").value += 'Nd';}
function Pm_Button(){document.getElementById("compounds").value += 'Pm';}
function Sm_Button(){document.getElementById("compounds").value += 'Sm';}
function Eu_Button(){document.getElementById("compounds").value += 'Eu';}
function Gd_Button(){document.getElementById("compounds").value += 'Gd';}
function Tb_Button(){document.getElementById("compounds").value += 'Tb';}
function Dy_Button(){document.getElementById("compounds").value += 'Dy';}
function Ho_Button(){document.getElementById("compounds").value += 'Ho';}
function Er_Button(){document.getElementById("compounds").value += 'Er';}
function Tm_Button(){document.getElementById("compounds").value += 'Tm';}
function Yb_Button(){document.getElementById("compounds").value += 'Yb';}
function Lu_Button(){document.getElementById("compounds").value += 'Lu';}
function Th_Button(){document.getElementById("compounds").value += 'Th';}
function Pa_Button(){document.getElementById("compounds").value += 'Pa';}
function U_Button(){document.getElementById("compounds").value += 'U';}
function Np_Button(){document.getElementById("compounds").value += 'Np';}
function Pu_Button(){document.getElementById("compounds").value += 'Pu';}