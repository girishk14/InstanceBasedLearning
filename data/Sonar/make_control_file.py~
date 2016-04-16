import json

def make_control_file_Arry():
	cf = open('control.json' , 'w')	
	metadata = {}
	metadata['location'] = ['data/Sonar/sonar.all-data']
	metadata['class_name'] = 'Class'
	metadata['attr_names'] = ['N/A'] * 60
	metadata['attr_types'] = ['c'] * 60
	metadata['class_position'] = 60
	cf.write(json.dumps(metadata, indent=1))

make_control_file_Arry()
