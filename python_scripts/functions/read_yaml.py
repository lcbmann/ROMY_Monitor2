def __readYaml(path, filename):
	'''
	read a yaml file
	'''

	import yaml


	with open(path+filename,'r') as file:
		stream = yaml.load(file, Loader=yaml.FullLoader)
		file.close()

	return stream
