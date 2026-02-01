import subprocess
import numpy

# Parameter yang diatur
sample_time = 1000 
span = 150  
step = 1  
min_overlap = 10  
threshold = 0.5  

# Fungsi untuk menghitung fingerprint dari file audio
def calculate_fingerprints(filename):
    fpcalc_out = subprocess.getoutput(f'fpcalc -raw -length {sample_time} {filename}')
    fingerprint_index = fpcalc_out.find('FINGERPRINT=') + 12
    fingerprints = list(map(int, fpcalc_out[fingerprint_index:].split(',')))      
    return fingerprints  

# Fungsi untuk menghitung korelasi antar dua daftar fingerprint
def correlation(listx, listy):
    if len(listx) == 0 or len(listy) == 0:    
        raise Exception('Empty lists cannot be correlated.')    
    if len(listx) > len(listy):     
        listx = listx[:len(listy)]  
    elif len(listx) < len(listy):       
        listy = listy[:len(listx)]      

    covariance = 0  
    for i in range(len(listx)):     
        covariance += 32 - bin(listx[i] ^ listy[i]).count("1")  
    covariance = covariance / float(len(listx))     
    return covariance / 32  

# Fungsi untuk menghitung cross-correlation antara dua daftar fingerprint dengan offset
def cross_correlation(listx, listy, offset):    
    if offset > 0:      
        listx = listx[offset:]      
        listy = listy[:len(listx)]  
    elif offset < 0:        
        offset = -offset        
        listy = listy[offset:]      
        listx = listx[:len(listy)]  
    if min(len(listx), len(listy)) < min_overlap:            
        return None   
    return correlation(listx, listy)  

# Fungsi untuk membandingkan dua daftar fingerprint dengan cross-correlation
def compare(listx, listy, span, step):  
    span = min(span, len(listx), len(listy))

    if span > min(len(listx), len(listy)):
        raise Exception('span >= sample size: %i >= %i\n' % (span, min(len(listx), len(listy))) + 'Reduce span, reduce crop or increase sample_time.')

    corr_xy = []    
    for offset in numpy.arange(-span, span + 1, step):
        corr_value = cross_correlation(listx, listy, offset)
        if corr_value is not None:  
            corr_xy.append(corr_value)
    
    return corr_xy  

# Fungsi untuk mendapatkan indeks dengan korelasi maksimal
def max_index(listx):   
    max_index = 0   
    max_value = listx[0]    
    for i, value in enumerate(listx):       
        if value > max_value:           
            max_value = value           
            max_index = i   
    return max_index  

# Fungsi untuk mendapatkan korelasi maksimal dan offsetnya
def get_max_corr(corr, source, target): 
    if not corr:  
        return "No valid correlation found."

    max_corr_index = max_index(corr)    
    max_corr_offset = -span + max_corr_index * step 
    print("max_corr_index = ", max_corr_index, "max_corr_offset = ", max_corr_offset)
    
    if corr[max_corr_index] > threshold:        
        return '%s and %s match with correlation of %.4f at offset %i' % (source, target, corr[max_corr_index], max_corr_offset) 
    else:
        return "No significant match found."

# Fungsi untuk melakukan korelasi antara dua file audio
def correlate(source, target):  
    fingerprint_source = calculate_fingerprints(source) 
    fingerprint_target = calculate_fingerprints(target)     
    corr = compare(fingerprint_source, fingerprint_target, span, step)  
    max_corr_offset = get_max_corr(corr, source, target)  

    return max_corr_offset
