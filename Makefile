all: comp vert frag

comp:
	glslangValidator -V src/shaders/shader.comp && mv comp.spv src/shaders/comp.spv

vert:
	glslangValidator -v src/shaders/shader.vert & mv vert.spv src/shaders/vert.spv

frag:
	glslangValidator -v src/shaders/shader.frag && mv frag.spv src/shaders/frag.spv