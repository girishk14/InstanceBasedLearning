import json

def make_control_file_Arry():
	cf = open('control.json' , 'w')	
	metadata = {}
	metadata['location'] = ['data/Image/seg.data']
	metadata['class_name'] = 'Class'
	metadata['attr_names'] = ['N/A'] * 19
	metadata['attr_types'] = ['c'] * 19
	metadata['class_position'] = 0
	cf.write(json.dumps(metadata, indent=1))

make_control_file_Arry()
