import json

def make_control_file_Arry():
	cf = open('control.json' , 'w')	
	metadata = {}
	metadata['location'] = ['data/Arcene/arcene.data']
	metadata['class_name'] = 'Class'
	metadata['attr_names'] = ['N/A'] * 10000
	metadata['attr_types'] = ['c'] * 10000
	metadata['class_position'] = 10000
	cf.write(json.dumps(metadata, indent=1))

make_control_file_Arry()
