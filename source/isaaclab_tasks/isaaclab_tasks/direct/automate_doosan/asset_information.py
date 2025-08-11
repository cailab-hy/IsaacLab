# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


def get_plug_info(ASSET_ID):
    base_z_offset = 0.0
    grasp_scale = 3.0
    diameter = None
    grasp_calibration = [0, 0, 0] # x, y, z

    # === JS Shin   (00004~00133) =====
    if ASSET_ID == "00004":
        height = 0.07515
        base_z_offset = 0.00619

    elif ASSET_ID == "00007":
        height = 0.01980
        base_z_offset = 0.02010

    elif ASSET_ID == "00014":
        height = 0.04532
        base_z_offset = 0.01365
        grasp_scale = 3.3

    elif ASSET_ID == "00015":
        height = 0.04307
        grasp_scale = 2.45

    elif ASSET_ID == "00016":
        height = 0.05158
        grasp_scale = 2.6
        grasp_calibration = [-0.00032, 0.00380, 0]

    elif ASSET_ID == "00021":
        height = 0.04357

    elif ASSET_ID == "00028":
        height = 0.06000

    elif ASSET_ID == "00030":
        height = 0.05345

    elif ASSET_ID == "00032":
        height = 0.02013
        base_z_offset = 0.01716

    elif ASSET_ID == "00042":
        height = 0.01979
        grasp_scale = 2.54

    elif ASSET_ID == "00062":
        height = 0.05639
        base_z_offset = 0.02596
        grasp_scale = 2.5

    elif ASSET_ID == "00074":
        height = 0.04220
        base_z_offset = 0.06617
        grasp_scale = 3.5

    elif ASSET_ID == "00077":
        height = 0.01026
        base_z_offset = 0.02415

    elif ASSET_ID == "00078":
        height = 0.05864
        base_z_offset = 0.00795
        grasp_scale = 2.7

    elif ASSET_ID == "00081":
        height = 0.01835
        base_z_offset = 0.02127

    elif ASSET_ID == "00083":
        height = 0.03440
        base_z_offset = 0.00697
        grasp_scale = 2.8

    elif ASSET_ID == "00103":
        height = 0.08289
        grasp_scale = 2.4

    elif ASSET_ID == "00110":
        height = 0.07586
        grasp_scale = 2.6

    elif ASSET_ID == "00117":
        height = 0.11124
        diameter = 0.00405

    elif ASSET_ID == "00133":
        height = 0.04371
        grasp_scale = 2.55
        grasp_calibration = [0, -0.00398, 0]

    # === HG Park   (00138~00318) =====    
    elif ASSET_ID == "00138":            
        height = 0.03379

    elif ASSET_ID == "00141":               
        height = 0.04286

    elif ASSET_ID == "00143":             
        height = 0.06427
        diameter = 0.00527
        grasp_scale = 3.25
        grasp_calibration = [-0.00470, -0.00196, 0]

    elif ASSET_ID == "00163":              
        height = 0.03504

    elif ASSET_ID == "00175":            
        height = 0.06115

    elif ASSET_ID == "00186":             
        height = 0.01349

    elif ASSET_ID == "00187":            
        height = 0.05958

    elif ASSET_ID == "00190":             
        height = 0.01267
        base_z_offset = 0.05832

    elif ASSET_ID == "00192":              
        height = 0.07553
        base_z_offset = 0.02664
        grasp_scale = 2.75
        grasp_calibration = [0, 0.00264, 0]

    elif ASSET_ID == "00210":             
        height = 0.07215

    elif ASSET_ID == "00211":               
        height = 0.05956

    elif ASSET_ID == "00213":              
        height = 0.04998
        diameter = 0.00444
        grasp_scale = 2.5
        grasp_calibration = [-0.00120, 0, 0]

    elif ASSET_ID == "00255":            
        height = 0.02317
        base_z_offset = 0.02032

    elif ASSET_ID == "00256":             
        height = 0.03524

    elif ASSET_ID == "00271":             
        height = 0.04441
        base_z_offset = 0.00258

    elif ASSET_ID == "00293":              
        height = 0.04972

    elif ASSET_ID == "00296":              
        height = 0.03963
        base_z_offset = 0.01873

    elif ASSET_ID == "00301":             
        height = 0.06832
        grasp_scale = 4.0

    elif ASSET_ID == "00308":             
        height = 0.01292
        base_z_offset = 0.01375

    elif ASSET_ID == "00318":            
        height = 0.07458
        grasp_scale = 2.75 
        
    # === WW Park   (00319~00499) =====
    elif ASSET_ID == "00319":
        height = 0.07849

    elif ASSET_ID == "00320":
        height = 0.07603

    elif ASSET_ID == "00329":
        height = 0.03608
        grasp_scale = 2.4

    elif ASSET_ID == "00340":
        height = 0.0316
        base_z_offset = 0.01469

    elif ASSET_ID == "00345":
        height = 0.05971

    elif ASSET_ID == "00346":
        height = 0.04918
        grasp_scale = 2.5

    elif ASSET_ID == "00360":
        height = 0.05846
        grasp_scale = 3.3

    elif ASSET_ID == "00388":
        height = 0.04453

    elif ASSET_ID == "00410":
        height = 0.04385

    elif ASSET_ID == "00417":
        height = 0.03996
        grasp_scale = 2.7

    elif ASSET_ID == "00422":
        height = 0.01280
        base_z_offset = 0.01788

    elif ASSET_ID == "00426":
        height = 0.04594
        base_z_offset = 0.01638
        grasp_scale = 2.7

    elif ASSET_ID == "00437":
        height = 0.04232

    elif ASSET_ID == "00444":
        height = 0.02195
        base_z_offset = 0.00739
        diameter = 0.01208
        grasp_scale = 2.5
        grasp_calibration = [0, 0.0085, 0]

    elif ASSET_ID == "00446":
        height = 0.06438
        base_z_offset = 0.00172

    elif ASSET_ID == "00470":
        height = 0.02527
        base_z_offset = 0.01266
        diameter = 0.02173
        grasp_calibration = [0.002, 0, 0]

    elif ASSET_ID == "00471":
        height = 0.02968
        grasp_scale = 3.3

    elif ASSET_ID == "00480":
        height = 0.03759

    elif ASSET_ID == "00486":
        height = 0.064
        base_z_offset = 0.00823

    elif ASSET_ID == "00499":
        height = 0.06575
        grasp_scale = 2.6

    # === BC Kim    (00506~00731) =====
    elif ASSET_ID == "00506":
        height = 0.07546

    elif ASSET_ID == "00514":
        height = 0.05203
        grasp_scale = 2.7

    elif ASSET_ID == "00537":
        height = 0.07859
        base_z_offset = 0.00645

    elif ASSET_ID == "00553":
        height = 0.00979
        base_z_offset = 0.00984

    elif ASSET_ID == "00559":
        height = 0.05747
        grasp_scale = 2.5

    elif ASSET_ID == "00581":
        height = 0.00754
        base_z_offset = 0.00542

    elif ASSET_ID == "00597":
        height = 0.05001
        grasp_scale = 2.6

    elif ASSET_ID == "00614":
        height = 0.07395
        grasp_scale = 2.7

    elif ASSET_ID == "00615":
        height = 0.02844
        base_z_offset = 0.02779

    elif ASSET_ID == "00638":
        height = 0.06023
        grasp_scale = 2.7

    elif ASSET_ID == "00648":
        height = 0.09197
        grasp_scale = 2.7

    elif ASSET_ID == "00649":
        height = 0.04334

    elif ASSET_ID == "00652":
        height = 0.03462
        base_z_offset = 0.01676

    elif ASSET_ID == "00659":
        height = 0.04203

    elif ASSET_ID == "00681":
        height = 0.05752
        base_z_offset = 0.02401
        grasp_scale = 2.7
        grasp_calibration = [0, 0, 0.01]

    elif ASSET_ID == "00686":
        height = 0.03381

    elif ASSET_ID == "00700":
        height = 0.04602

    elif ASSET_ID == "00703":
        height = 0.04663
        diameter = 0.00917
        grasp_scale = 2.4

    elif ASSET_ID == "00726":
        height = 0.0319

    elif ASSET_ID == "00731":
        height = 0.04021
        
    # === SY Hong   (00741~01136) =====
    elif ASSET_ID == "00741":
        height = 0.00864
        base_z_offset = 0.01825

    elif ASSET_ID == "00755":
        height = 0.05290
        diameter = 0.01400
        grasp_scale = 2.8

    elif ASSET_ID == "00768":
        height = 0.01367
        base_z_offset = 0.03409

    elif ASSET_ID == "00783":
        height = 0.05131

    elif ASSET_ID == "00831":
        height = 0.05746

    elif ASSET_ID == "00855":
        height = 0.06480

    elif ASSET_ID == "00860":
        height = 0.05000

    elif ASSET_ID == "00863":
        height = 0.01675
        base_z_offset = 0.03140

    elif ASSET_ID == "01026":
        height = 0.02777
        base_z_offset = 0.01384
        grasp_calibration = [-0.0065, 0, 0]

    elif ASSET_ID == "01029":
        height = 0.05593
        grasp_scale = 2.6
        grasp_calibration = [0.0035, 0, 0]

    elif ASSET_ID == "01036":
        height = 0.04660
        diameter = 0.00100
        grasp_scale = 2.5

    elif ASSET_ID == "01041":
        height = 0.03295
        grasp_scale = 2.8

    elif ASSET_ID == "01053":
        height = 0.04380
        base_z_offset = 0.01360

    elif ASSET_ID == "01079":
        height = 0.03528
        base_z_offset = 0.00844
        grasp_scale = 2.8
    
    elif ASSET_ID == "01092":
        height = 0.03713
        base_z_offset = 0.00480
        grasp_calibration = [0, 0, 0.015]
        grasp_scale = 2.7
    
    elif ASSET_ID == "01102":
        height = 0.04465
    
    elif ASSET_ID == "01125":
        height = 0.02862
    
    elif ASSET_ID == "01129":
        height = 0.05033
        base_z_offset = 0.00562
    
    elif ASSET_ID == "01132":
        height = 0.09646
        grasp_scale = 2.8
    
    elif ASSET_ID == "01136":
        height = 0.05296

    return height, base_z_offset, diameter, grasp_scale, grasp_calibration


def get_socket_info(ASSET_ID):
    base_height = 0.0

    # === JS Shin   (00004~00133) =====
    if ASSET_ID == "00004":
        height = 0.02228

    elif ASSET_ID == "00007":
        height = 0.04073

    elif ASSET_ID == "00014":
        height = 0.02332

    elif ASSET_ID == "00015":
        height = 0.02000
        base_height = 0.00538

    elif ASSET_ID == "00016":
        height = 0.02023

    elif ASSET_ID == "00021":
        height = 0.00810

    elif ASSET_ID == "00028":
        height = 0.02370

    elif ASSET_ID == "00030":
        height = 0.01273
        base_height = 0.00749

    elif ASSET_ID == "00032":
        height = 0.02533

    elif ASSET_ID == "00042":
        height = 0.00740

    elif ASSET_ID == "00062":
        height = 0.04885

    elif ASSET_ID == "00074":
        height = 0.08070

    elif ASSET_ID == "00077":
        height = 0.04501

    elif ASSET_ID == "00078":
        height = 0.04132

    elif ASSET_ID == "00081":
        height = 0.03884

    elif ASSET_ID == "00083":
        height = 0.01669

    elif ASSET_ID == "00103":
        height = 0.00663

    elif ASSET_ID == "00110":
        height = 0.02981

    elif ASSET_ID == "00117":
        height = 0.06904

    elif ASSET_ID == "00133":
        height = 0.00229
        base_height = 0.02230

    # === HG Park   (00138~00318) =====
    elif ASSET_ID == "00138":
        height = 0.00845
        base_height=0.00208

    elif ASSET_ID == "00141":
        height = 0.02586

    elif ASSET_ID == "00143":
        height = 0.01118

    elif ASSET_ID == "00163":
        height = 0.00899

    elif ASSET_ID == "00175":
        height = 0.04004

    elif ASSET_ID == "00186":
        height = 0.02993

    elif ASSET_ID == "00187":
        height = 0.01003
        base_height=0.00950

    elif ASSET_ID == "00190":
        height = 0.07726

    elif ASSET_ID == "00192":
        height = 0.05439

    elif ASSET_ID == "00210":
        height = 0.04680

    elif ASSET_ID == "00211":
        height = 0.01759

    elif ASSET_ID == "00213":
        height = 0.00444

    elif ASSET_ID == "00255":
        height = 0.04187

    elif ASSET_ID == "00256":
        height = 0.01343

    elif ASSET_ID == "00271":
        height = 0.02441

    elif ASSET_ID == "00293":
        height = 0.01411

    elif ASSET_ID == "00296":
        height = 0.02604

    elif ASSET_ID == "00301":
        height = 0.02365

    elif ASSET_ID == "00308":
        height = 0.04943

    elif ASSET_ID == "00318":
        height = 0.03992


    # === WW Park   (00319~00499) =====
    elif ASSET_ID == "00319":
        height = 0.03710

    elif ASSET_ID == "00320":
        height = 0.01299

    elif ASSET_ID == "00329":
        height = 0.02188

    elif ASSET_ID == "00340":
        height = 0.03784
        base_height =   0.01469

    elif ASSET_ID == "00345":
        height = 0.03178

    elif ASSET_ID == "00346":
        height = 0.0042

    elif ASSET_ID == "00360":
        height = 0.02193

    elif ASSET_ID == "00388":
        height = 0.01474

    elif ASSET_ID == "00410":
        height = 0.01941

    elif ASSET_ID == "00417":
        height = 0.01520

    elif ASSET_ID == "00422":
        height = 0.02920
        base_height = 0.01788

    elif ASSET_ID == "00426":
        height = 0.0266

    elif ASSET_ID == "00437":
        height = 0.01426
        base_height =0.00751

    elif ASSET_ID == "00444":
        height = 0.01556

    elif ASSET_ID == "00446":
        height = 0.0605

    elif ASSET_ID == "00470":
        height = 0.01846

    elif ASSET_ID == "00471":
        height = 0.00523
        

    elif ASSET_ID == "00480":
        height = 0.01271

    elif ASSET_ID == "00486":
        height = 0.04450
        base_height =0.00823

    elif ASSET_ID == "00499":
        height = 0.02979

    # === BC Kim    (00506~00731) =====
    elif ASSET_ID == "00506":
        height = 0.04783

    elif ASSET_ID == "00514":
        height = 0.01509

    elif ASSET_ID == "00537":
        height = 0.06487

    elif ASSET_ID == "00553":
        height = 0.03489

    elif ASSET_ID == "00559":
        height = 0.00829
        base_height = 0.00431

    elif ASSET_ID == "00581":
        height = 0.03153

    elif ASSET_ID == "00597":
        height = 0.0221 
        base_height = 0.00416

    elif ASSET_ID == "00614":
        height = 0.02311
        base_height = 0.0288

    elif ASSET_ID == "00615":
        height = 0.04739

    elif ASSET_ID == "00638":
        height = 0.03326

    elif ASSET_ID == "00648":
        height = 0.0689

    elif ASSET_ID == "00649":
        height = 0.00619

    elif ASSET_ID == "00652":
        height = 0.02317

    elif ASSET_ID == "00659":
        height = 0.00841

    elif ASSET_ID == "00681":
        height = 0.0422

    elif ASSET_ID == "00686":
        height = 0.0118

    elif ASSET_ID == "00700":
        height = 0.00642

    elif ASSET_ID == "00703":
        height = 0.017
        base_height = 0.01059

    elif ASSET_ID == "00726":
        height = 0.0321

    elif ASSET_ID == "00731":
        height = 0.00893
        base_height = 0.01031

    # === SY Hong   (00741~01136) =====
    elif ASSET_ID == "00741":
        height = 0.02715
    
    elif ASSET_ID == "00755":
        height = 0.00782
    
    elif ASSET_ID == "00768":
        height = 0.05825
    
    elif ASSET_ID == "00783":
        height = 0.01324
    
    elif ASSET_ID == "00831":
        height = 0.00575
    
    elif ASSET_ID == "00855":
        height = 0.02946
    
    elif ASSET_ID == "00860":
        height = 0.00968
    
    elif ASSET_ID == "00863":
        height = 0.04817
    
    elif ASSET_ID == "01026":
        height = 0.04163
    
    elif ASSET_ID == "01029":
        height = 0.03043
    
    elif ASSET_ID == "01036":
        height = 0.01036
    
    elif ASSET_ID == "01041":
        height = 0.00901
    
    elif ASSET_ID == "01053":
        height = 0.02909
    
    elif ASSET_ID == "01079":
        height = 0.01990
    
    elif ASSET_ID == "01092":
        height = 0.01704
    
    elif ASSET_ID == "01102": 
        height = 0.01456
    
    elif ASSET_ID == "01125":
        height = 0.00711
    
    elif ASSET_ID == "01129":
        height = 0.00478
        base_height = 0.00365
    
    elif ASSET_ID == "01132":
        height = 0.05682
    
    elif ASSET_ID == "01136":
        height = 0.01122

    return height, base_height
