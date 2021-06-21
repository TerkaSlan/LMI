from imports import np
from scipy.spatial.distance import cityblock

def get_cophir_distance(object_1, object_2):
    """ Implements messif.objects.impl.MetaObjectSAPIRWeightedDist2

    Parameters
    ----------
    object_1: np.array
        Descriptors of pivot
    
    object_2: np.array
        Descriptors of the object
    
    Returns
    -------
    Distance of query and an object combining all of the descriptors.
    """
    assert object_1.shape == object_2.shape and object_1.shape[0] == 282
    result = 1.5 * (get_color_layout_type(object_1, object_2) / 300) + \
             2.5 * (get_color_structure_type(object_1, object_2) / 40 / 255) + \
             4.5 * (get_edge_histogram_type(object_1, object_2) / 68) + \
             0.5 * (get_homogenous_texture_type(object_1, object_2) / 25) + \
             2.5 * (get_scalable_color_type(object_1, object_2) / 3000)

    return result

def get_color_layout_type(object_1, object_2):
    """ Implements messif.objects.impl.ObjectColorLayout

    Parameters
    ----------
    object_1: np.array
        Descriptors of pivot
    
    object_2: np.array
        Descriptors of the object
    
    Returns
    -------
    Distance of color layout portion of descriptors.
    """
    y_labels = [i for i in range(6)]
    cb_labels = [6+i for i in range(3)]
    cr_labels = [9+i for i in range(3)]
    y_coeff_query = object_1[y_labels]
    cb_coeff_query = object_1[cb_labels]
    cr_coeff_query = object_1[cr_labels]

    y_weights = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    cb_weights = [2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    cr_weights = [4, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    def sum_coeff(weights, coeffs1, coeffs2):
        rtv = 0
        for j in range(min(len(coeffs1), len(coeffs2))-1, -1, -1):
            diff = coeffs1[j] - coeffs2[j]
            rtv += weights[j]*diff*diff
        return rtv
            
    def get_distance_impl(object_2):
        y_coeff_object = object_2[y_labels]
        cb_coeff_object = object_2[cb_labels]
        cr_coeff_object = object_2[cr_labels]
        return np.sqrt(sum_coeff(y_weights, y_coeff_query, y_coeff_object)) + \
            np.sqrt(sum_coeff(cb_weights, cb_coeff_query, cb_coeff_object)) + \
            np.sqrt(sum_coeff(cr_weights, cr_coeff_query, cr_coeff_object))

    return get_distance_impl(object_2)

def get_color_structure_type(object_1, object_2):
    """ Implements messif.objects.impl.ObjectShortVectorL1

    Parameters
    ----------
    object_1: np.array
        Descriptors of pivot
    
    object_2: np.array
        Descriptors of the object
    
    Returns
    -------
    Distance of color structure type portion of descriptors.
    """
    query = object_1[12:12+64]
    obj = object_2[12:12+64]
    return cityblock(query, obj)

def get_edge_histogram_type(object_1, object_2):
    """ Implements messif.objects.impl.ObjectVectorEdgecomp

    Parameters
    ----------
    object_1: np.array
        Descriptors of pivot
    
    object_2: np.array
        Descriptors of the object
    
    Returns
    -------
    Distance of edge histogram type portion of descriptors.
    """
    edge_query = object_1[12+64:12+64+80]
    edge_object = object_2[12+64:12+64+80]

    quanttable = [
        [0.010867,0.057915,0.099526,0.144849,0.195573,0.260504,0.358031,0.530128],
        [0.012266,0.069934,0.125879,0.182307,0.243396,0.314563,0.411728,0.564319],
        [0.004193,0.025852,0.046860,0.068519,0.093286,0.123490,0.161505,0.228960],
        [0.004174,0.025924,0.046232,0.067163,0.089655,0.115391,0.151904,0.217745],
        [0.006778,0.051667,0.108650,0.166257,0.224226,0.285691,0.356375,0.450972],
    ]

    def EHD_Make_Global_SemiGlobal(totalHistogram):
        for i in range(5): 
            totalHistogram[i] = 0
        for j in range(0, 80, 5):
            for i in range(5):
                totalHistogram[i] += totalHistogram[5+i+j]
        for i in range(5):
            totalHistogram[i] = totalHistogram[i]*5/16
            
        for i in range(85,105,1):
            j = i-85
            totalHistogram[i] = \
                    (totalHistogram[5+j] \
                    +totalHistogram[5+20+j] \
                    +totalHistogram[5+40+j] \
                    +totalHistogram[5+60+j])/4
        
        for i in range(105,125,1):
            j = i-105
            totalHistogram[i] = \
                    (totalHistogram[5+20*(j//5)+j%5] \
                    +totalHistogram[5+20*(j//5)+j%5+5] \
                    +totalHistogram[5+20*(j//5)+j%5+10] \
                    +totalHistogram[5+20*(j//5)+j%5+15])/4
            
        for i in range(125,135,1):
            j = i-125
            totalHistogram[i] = \
                    (totalHistogram[5+10*(j//5)+0+j%5] \
                    +totalHistogram[5+10*(j//5)+5+j%5] \
                    +totalHistogram[5+10*(j//5)+20+j%5] \
                    +totalHistogram[5+10*(j//5)+25+j%5])/4

        for i in range(135,145,1):
            j = i-135
            totalHistogram[i] = \
                    (totalHistogram[5+10*(j//5)+40+j%5] \
                    +totalHistogram[5+10*(j//5)+45+j%5] \
                    +totalHistogram[5+10*(j//5)+60+j%5] \
                    +totalHistogram[5+10*(j//5)+65+j%5])/4
            
        for i in range(145,150,1):
            j = i-145
            totalHistogram[i] = \
                    (totalHistogram[5+25+j%5] \
                    +totalHistogram[5+30+j%5] \
                    +totalHistogram[5+45+j%5] \
                    +totalHistogram[5+50+j%5])/4

    total_edgehist_ref = [0 for i in range(150)]; total_edgehist_query = [0 for i in range(150)]
    
    for i in range(80):
        total_edgehist_ref[i+5] = quanttable[i%5][edge_query[i]]
        total_edgehist_query[i+5] = quanttable[i%5][edge_object[i]]
    
    EHD_Make_Global_SemiGlobal(total_edgehist_ref)
    EHD_Make_Global_SemiGlobal(total_edgehist_query)
    
    dist = 0
    for i in range(80+70):
        dTemp = abs(total_edgehist_ref[i] - total_edgehist_query[i])
        dist += dTemp
    return dist

def get_homogenous_texture_type(object_1, object_2):
    """ Implements messif.objects.impl.ObjectHomogeneousTexture

    Parameters
    ----------
    object_1: np.array
        Descriptors of pivot
    
    object_2: np.array
        Descriptors of the object
    
    Returns
    -------
    Distance of homogenous texture type portion of descriptors.
    """
    texture_query = object_1[12+64+80:12+64+80+62]
    texture_object = object_2[12+64+80:12+64+80+62]

    NUMofFEATURE = 62
    Quant_level = 255
    RadialDivision = 5
    AngularDivision = 6
    wm=[0.42,1.00,1.00,0.08,1.00]
    wd=[0.32,1.00,1.00,1.00,1.00]
    wdc=0.28
    wstd=0.22
    temp_distance = 0; distance=0
    RefFeature = [0 for i in range(NUMofFEATURE)]
    QueryFeature = [0 for i in range(NUMofFEATURE)]
    fRefFeature = [0 for i in range(NUMofFEATURE)]
    fQueryFeature = [0 for i in range(NUMofFEATURE)]

    def object_homogenous_texture(texture_query):
        semicol = [1,1,30,30]
        average = texture_query[0]
        std = texture_query[1]
        energy = texture_query[sum(semicol[:2]):sum(semicol[:3])]
        energy_deviation = texture_query[sum(semicol[:3]):sum(semicol[:4])]
        return average, std, energy, energy_deviation
    
    q_average, q_std, q_energy, q_energy_d = object_homogenous_texture(texture_query)
    o_average, o_std, o_energy, o_energy_d = object_homogenous_texture(texture_object)

    RefFeature[0] = o_average
    RefFeature[1] = o_std
    QueryFeature[0] = q_average
    QueryFeature[1] = q_std

    dcmin=0.0; dcmax=255.0; stdmin=1.309462; stdmax=109.476530

    mmax= [[18.392888,18.014313,18.002143,18.083845,18.046575,17.962099],
        [19.368960,18.628248,18.682786,19.785603,18.714615,18.879544],
        [20.816939,19.093605,20.837982,20.488190,20.763511,19.262577],
        [22.298871,20.316787,20.659550,21.463502,20.159304,20.280403],
        [21.516125,19.954733,20.381041,22.129800,20.184864,19.999331]]

    mmin= [[ 6.549734, 8.886816, 8.885367, 6.155831, 8.810013, 8.888925],
        [ 6.999376, 7.859269, 7.592031, 6.754764, 7.807377, 7.635503],
        [ 8.299334, 8.067422, 7.955684, 7.939576, 8.518458, 8.672599],
        [ 9.933642, 9.732479, 9.725933, 9.802238,10.076958,10.428015],
        [11.704927,11.690975,11.896972,11.996963,11.977944,11.944282]]

    dmax=[ [21.099482,20.749788,20.786944,20.847705,20.772294,20.747129],
        [22.658359,21.334119,21.283285,22.621111,21.773690,21.702166],
        [24.317046,21.618960,24.396872,23.797967,24.329333,21.688523],
        [25.638742,24.102725,22.687910,25.216958,22.334769,22.234942],
        [24.692990,22.978804,23.891302,25.244315,24.281915,22.699811]]

    dmin=[[ 9.052970,11.754891,11.781252, 8.649997,11.674788,11.738701],
        [ 9.275178,10.386329,10.066189, 8.914539,10.292868,10.152977],
        [10.368594,10.196313,10.211122,10.112823,10.648101,10.801070],
        [11.737487,11.560674,11.551509,11.608201,11.897524,12.246614],
        [13.303207,13.314553,13.450340,13.605001,13.547492,13.435994]]

    def HTDequantization(intFeature, floatFeature):
        dcstep=(dcmax-dcmin)/Quant_level
        floatFeature[0]=(dcmin+intFeature[0]*dcstep)
        stdstep=(stdmax-stdmin)/Quant_level
        floatFeature[1]=(stdmin+intFeature[1]*stdstep)
        
        for n in range(RadialDivision):
            for m in range(AngularDivision):
                mstep=(mmax[n][m]-mmin[n][m])/Quant_level
                floatFeature[n*AngularDivision+m+2]=(mmin[n][m]+intFeature[n*AngularDivision+m+2]*mstep)
        for n in range(RadialDivision):
            for m in range(AngularDivision):
                dstep=(dmax[n][m]-dmin[n][m])/Quant_level
                floatFeature[n*AngularDivision+m+32]=(dmin[n][m]+intFeature[n*AngularDivision+m+32]*dstep)
        return intFeature, floatFeature

    dcnorm=122.331353
    stdnorm=51.314701
    mmean= [[13.948462, 15.067986, 15.077915, 13.865536, 15.031283, 15.145633],
    [15.557970, 15.172251, 15.357618, 15.166167, 15.414601, 15.414378],
    [17.212408, 16.173027, 16.742651, 16.913837, 16.911480, 16.582123],
    [17.911104, 16.761711, 17.065447, 17.867548, 17.250889, 17.050728],
    [17.942741, 16.891190, 17.101770, 18.032434, 17.295305, 17.202160]]
    dmean= [[16.544933, 17.845844, 17.849176, 16.484509, 17.803377, 17.928810],
    [18.054886, 17.617800, 17.862095, 17.627794, 17.935352, 17.887453],
    [19.771456, 18.512341, 19.240444, 19.410559, 19.373478, 18.962496],
    [20.192045, 18.763544, 19.202494, 20.098207, 19.399082, 19.032280],
    [19.857040, 18.514065, 18.831860, 19.984838, 18.971045, 18.863575]]

    def HTNormalization(feature):
        feature[0]/=dcnorm
        feature[1]/=stdnorm
        
        for n in range(RadialDivision):
            for m in range(AngularDivision):
                feature[n*AngularDivision+m+2]/=mmean[n][m]
        for n in range(RadialDivision):
            for m in range(AngularDivision):
                feature[n*AngularDivision+m+32]/=dmean[n][m]
        return feature

    for i in range(30):
        RefFeature[i+2] = o_energy[i]
        QueryFeature[i+2] = q_energy[i]
        RefFeature[i+30+2] = o_energy_d[i]
        QueryFeature[i+30+2] = q_energy_d[i]

        
    RefFeature, fRefFeature = HTDequantization(RefFeature, fRefFeature)
    QueryFeature,fQueryFeature = HTDequantization(QueryFeature,fQueryFeature)
    fRefFeature = HTNormalization(fRefFeature)
    fQueryFeature = HTNormalization(fQueryFeature)

    distance = (wdc*abs(fRefFeature[0]-fQueryFeature[0]))
    distance +=(wstd*abs(fRefFeature[1]-fQueryFeature[1]))

    min = 100000.00
    for j in range(3):
        for i in range(AngularDivision, 0, -1):
            temp_distance =0.0
            for n in range(2, RadialDivision):
                for m in range(AngularDivision):
                    if m >= i:
                        temp_distance+=(wm[n]*abs(fRefFeature[n*AngularDivision+m+2-i]-fQueryFeature[(n-j)*AngularDivision+m+2])) + (wd[n]*abs(fRefFeature[n*AngularDivision+m+30+2-i]-fQueryFeature[(n-j)*AngularDivision+m+30+2]))
                    else:
                        temp_distance+=(wm[n]*abs(fRefFeature[(n+1)*AngularDivision+m+2-i]-fQueryFeature[(n-j)*AngularDivision+m+2])) + (wd[n]*abs(fRefFeature[(n+1)*AngularDivision+m+30+2-i]-fQueryFeature[(n-j)*AngularDivision+m+30+2]))
            if (temp_distance < min): min = temp_distance
        
    for j in range(1,3):
        for i in range(AngularDivision, 0, -1):
            temp_distance =0.0
            for n in range(2, RadialDivision):
                for m in range(AngularDivision):
                    if m >= i:
                        temp_distance+=(wm[n]*abs(fRefFeature[(n-j)*AngularDivision+m+2-i]-fQueryFeature[n*AngularDivision+m+2])) + (wd[n]*abs(fRefFeature[(n-j)*AngularDivision+m+30+2-i]-fQueryFeature[n*AngularDivision+m+30+2]))
                    else:
                        temp_distance+=(wm[n]*abs(fRefFeature[(n+1-j)*AngularDivision+m+2-i]-fQueryFeature[n*AngularDivision+m+2]))+ (wd[n]*abs(fRefFeature[(n+1-j)*AngularDivision+m+30+2-i]-fQueryFeature[n*AngularDivision+m+30+2]))
            if (temp_distance < min): min = temp_distance
    distance = min + distance
    return distance

def get_scalable_color_type(object_1, object_2):
    """ Implements messif.objects.impl.ObjectIntVectorL1

    Parameters
    ----------
    object_1: np.array
        Descriptors of pivot
    
    object_2: np.array
        Descriptors of the object
    
    Returns
    -------
    Distance of scalable color type portion of descriptors.
    """

    query = object_1[12+64+80+62:12+64+80+62+64]
    obj = object_2[12+64+80+62:12+64+80+62+64]
    return cityblock(query, obj)

def get_test_cophir_distances():
    """ Ground truth of meta-distance of cophir, obtained from java implementation.

    Returns
    -------
    Array of distances and relevant object_id. Pivot id is 296166
    """
    return np.array([
        [1.6351128, 0.18779817, 0.14843136, 0.1594384, 0.07139549, 0.09166667, 51212727.0],
        [1.682718, 0.09080713, 0.13764705, 0.17387441, 0.10157666, 0.14766666, 25883336.0],
        [1.6881065, 0.1396715, 0.16705881, 0.15252937, 0.069139846, 0.136, 4957200.0],
        [1.6905662, 0.14340712, 0.09480392, 0.19853693, 0.11005905, 0.116, 92783215.0],
        [1.6972493, 0.112000085, 0.18303922, 0.15432863, 0.08101115, 0.13466667, 70408140.0],
        [1.7225393, 0.22863331, 0.1427451, 0.13901977, 0.074275084, 0.144, 4002329.0],
        [1.7333848, 0.12465594, 0.15941177, 0.14752053, 0.09639171, 0.17433333, 25884293.0],
        [1.7412349, 0.10851099, 0.16411763, 0.14952573, 0.085617, 0.181, 95212774.0],
        [1.7468263, 0.16563658, 0.1509804, 0.16546871, 0.070955776, 0.13633333, 25884855.0],
        [1.7601067, 0.116973855, 0.20107844, 0.1379662, 0.052203927, 0.174, 26993235.0],
        [1.7687613, 0.14129323, 0.17882353, 0.14637578, 0.06047651, 0.16833334, 25876064.0],
        [1.7776502, 0.17072561, 0.17803922, 0.15085828, 0.05186963, 0.14866666, 37220387.0],
        [1.780572, 0.12615824, 0.14931373, 0.1746563, 0.064194046, 0.16, 25884364.0],
        [1.7837722, 0.15694995, 0.16019607, 0.17438854, 0.06121737, 0.133, 36113013.0],
        [1.7900904, 0.12776974, 0.1877451, 0.16944546, 0.07147057, 0.13233334, 70408193.0],
        [1.790114, 0.1596585, 0.15254903, 0.18508218, 0.061101142, 0.12233333, 36114491.0],
        [1.8026476, 0.12207319, 0.16215685, 0.16443859, 0.06834421, 0.176, 25884306.0],
        [1.8031311, 0.11948798, 0.15431371, 0.17523155, 0.069146015, 0.166, 25875549.0],
        [1.8108573, 0.1495503, 0.20166667, 0.13175838, 0.117238246, 0.17233333, 25879946.0],
        [1.813886, 0.13701326, 0.17911765, 0.18313913, 0.051225066, 0.12433334, 70407919.0],
        [1.8195415, 0.12606376, 0.14186274, 0.1851772, 0.06831647, 0.16333333, 25877568.0],
        [1.8233508, 0.10657194, 0.1822549, 0.17682712, 0.06593359, 0.15166667, 25883458.0],
        [1.8268771, 0.08052375, 0.20813726, 0.16621116, 0.07392963, 0.16033334, 47056819.0],
        [1.8270887, 0.16441211, 0.16137256, 0.17175215, 0.05664223, 0.15033333, 16611982.0],
        [1.8273013, 0.12164973, 0.18745098, 0.15821242, 0.07015344, 0.17166667, 36112431.0],
        [1.8274928, 0.14421116, 0.14254901, 0.1880133, 0.09415405, 0.14466667, 25880963.0],
        [1.8346976, 0.14509463, 0.15009804, 0.19059214, 0.1032919, 0.133, 36115099.0],
        [1.8351744, 0.19697264, 0.14303921, 0.16870122, 0.1259239, 0.144, 36115353.0],
        [1.836216, 0.15051389, 0.22911765, 0.14311036, 0.058975324, 0.14566667, 23311992.0]])

def get_test_color_layout_type():
    distances = get_test_cophir_distances()
    return np.array([distances[:, 1], distances[:, -1]]).T

def get_test_color_structure_type():
    distances = get_test_cophir_distances()
    return np.array([distances[:, 2], distances[:, -1]]).T

def get_test_edge_histogram_type():
    distances = get_test_cophir_distances()
    return np.array([distances[:, 3], distances[:, -1]]).T

def get_test_homogenous_texture_type():
    distances = get_test_cophir_distances()
    return np.array([distances[:, 4], distances[:, -1]]).T

def get_test_scalable_color_type():
    distances = get_test_cophir_distances()
    return np.array([distances[:, 5], distances[:, -1]]).T

def get_test_all():
    distances = get_test_cophir_distances()
    return np.array([distances[:, 0], distances[:, -1]]).T