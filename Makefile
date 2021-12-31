shader:
	glslangValidator -V src/shaders/shader.comp && mv comp.spv src/shaders/comp.spv
