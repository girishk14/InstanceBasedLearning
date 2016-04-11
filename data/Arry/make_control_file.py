import json

def make_control_file_Arry():
	cf = open('control.json' , 'w')	
	metadata = {}
	metadata['location'] = ['arrhythmia.data']
	metadata['class_name'] = 'Class'
	metadata['attr_names'] = ['N/A'] * 279
	metadata['attr_types'] = ['c'] * 279
	metadata['class_position'] = 279
	cf.write(json.dumps(metadata, indent=1))

make_control_file_Arry()
