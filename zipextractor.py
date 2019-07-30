# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 18:44:00 2018

@author: SusarlaS
"""

import re
# Create a variable containing a text string
text = '00045-'

re.findall(r'[0-9]{5}(?:-[0-9]{4})?', text)

zipcode_zipext = re.findall(r'[0-9]{5}(?:-[0-9]{4})?', text)
zipcode =  zipcode_zipext.split('-')

