#shader vertex
#version 330 core

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 velocity;
layout(location = 3) in float type;

out vec4 geom_vel;
out float geom_type;

void main()
{
    gl_Position =  position;
	geom_vel = velocity;
	geom_type = type;
};

###################################################

#shader geometry
#version 330 core
layout(points) in;
layout(triangle_strip, max_vertices = 12) out;

in vec4 geom_vel[];
in float geom_type[];

flat out float fragment_type;

mat4 R(const float angle) {
	return mat4(sin(angle), -cos(angle), 0, 0,
		cos(angle), sin(angle), 0, 0,
		0, 0, 0, 0, 
		0, 0, 0, 0);
}

vec4 getPosition(vec4 offset){
	float angle = atan(geom_vel[0].y, geom_vel[0].x);
	offset = R(angle)*offset;
	return gl_in[0].gl_Position + offset;
}

void main() {
	fragment_type = geom_type[0];

	if (fragment_type == 1)
	{
		gl_Position = getPosition(vec4(-0.01, 0.0, 0.0, 0.0));
		EmitVertex();
		gl_Position = getPosition(vec4(0.01, 0.0, 0.0, 0.0));
		EmitVertex();
		gl_Position = getPosition(vec4(0.0, 0.01, 0.0, 0.0));
		EmitVertex();
		EndPrimitive();

		gl_Position = getPosition(vec4(-0.004, 0.0, 0.0, 0.0));
		EmitVertex();
		gl_Position = getPosition(vec4(0.004, 0.0, 0.0, 0.0));
		EmitVertex();
		gl_Position = getPosition(vec4(-0.004, -0.032, 0.0, 0.0));
		EmitVertex();
		gl_Position = getPosition(vec4(0.004, -0.032, 0.0, 0.0));
		EmitVertex();
		EndPrimitive();
	}
	else if (fragment_type == 2)
	{
		gl_Position = getPosition(vec4(-0.02, 0.0, 0.0, 0.0));
		EmitVertex();
		gl_Position = getPosition(vec4(0.02, 0.0, 0.0, 0.0));
		EmitVertex();
		gl_Position = getPosition(vec4(0.0, 0.02, 0.0, 0.0));
		EmitVertex();
		EndPrimitive();

		gl_Position = getPosition(vec4(-0.015, 0.0, 0.0, 0.0));
		EmitVertex();
		gl_Position = getPosition(vec4(0.015, 0.0, 0.0, 0.0));
		EmitVertex();
		gl_Position = getPosition(vec4(-0.015, -0.040, 0.0, 0.0));
		EmitVertex();
		gl_Position = getPosition(vec4(0.015, -0.040, 0.0, 0.0));
		EmitVertex();
		EndPrimitive();
	}
	else
	{
		gl_Position = getPosition(vec4(-0.016, 0.0, 0.0, 0.0));
		EmitVertex();
		gl_Position = getPosition(vec4(0.016, 0.0, 0.0, 0.0));
		EmitVertex();
		gl_Position = getPosition(vec4(0.0, 0.02, 0.0, 0.0));
		EmitVertex();
		EndPrimitive();

		gl_Position = getPosition(vec4(-0.004, 0.0, 0.0, 0.0));
		EmitVertex();
		gl_Position = getPosition(vec4(0.004, 0.0, 0.0, 0.0));
		EmitVertex();
		gl_Position = getPosition(vec4(-0.004, -0.016, 0.0, 0.0));
		EmitVertex();
		gl_Position = getPosition(vec4(0.004, -0.016, 0.0, 0.0));
		EmitVertex();
		EndPrimitive();
	}
}

##################################################

#shader fragment
#version 330 core
layout(location = 0) out vec4 color;

flat in float fragment_type;

void main()
{
	if (fragment_type == 1)
	{
		color = vec4(0.2f, 0.9f, 0.7f, 1.0f);
	}
	else if (fragment_type == 2)
	{
		color = vec4(0.5f, 0.4f, 0.5f, 1.0f);
	}
	else
	{
		color = vec4(0.8f, 0.3f, 0.8f, 1.0f);
	}
		
};

