import json

def make_control_file_Arry():
	cf = open('control.json' , 'w')	
	metadata = {}
	metadata['location'] = ['data/Libra/movement_libras.data']
	metadata['class_name'] = 'Class'
	metadata['attr_names'] = ['N/A'] * 90
	metadata['attr_types'] = ['c'] * 90
	metadata['class_position'] = 90
	cf.write(json.dumps(metadata, indent=1))

make_control_file_Arry()
