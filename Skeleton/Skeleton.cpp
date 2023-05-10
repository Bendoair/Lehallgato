//=============================================================================================
// Computer Graphics Sample Program: Ray-tracing-let
//=============================================================================================
#include "framework.h"

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd* M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Sphere : public Intersectable {
	vec3 center;
	float radius;

	Sphere(const vec3& _center, float _radius, Material* _material) {
		center = _center;
		radius = _radius;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0f;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) * (1.0f / radius);
		hit.material = material;
		return hit;
	}
};

struct Triangle :Intersectable{
	vec3 r1, r2, r3;
	vec3 norm;

	Triangle(vec3 ia, vec3 ib, vec3 ic, Material* mat) {
		r1 = ia; r2 = ib; r3 = ic;
		norm = normalize((r2-r1)*(r3-r1));
		material = mat;
	}

	Triangle() {
		r1 = r2 = r3 = norm = vec3(0, 0, 0);
	}


	Hit intersect(const Ray& ray){
		Hit hit;
		float t = dot((r1 - ray.start), norm) / dot(ray.dir, norm);
		vec3 p = ray.start + ray.dir * t;
		if (dot(cross((r2 - r1), (p - r1)), norm) > 0 
			&& dot(cross((r3 - r2), (p - r2)), norm) > 0 
			&& dot(cross((r1 - r3), (p - r3)), norm) > 0) {
			hit.position = p;
			hit.normal = norm;
			hit.material = material;
		}
		return hit;
	}
};

struct Side :Intersectable{
	vec3 a, b, c, d;
	vec3 norm;

	Side() {
		a =b=c=d= vec3(0, 0, 0);
		norm = vec3(0, 0, 0);
	}
	Side(vec3 a1, vec3 b1, vec3 c1, vec3 d1, Material* mat) {
		material = mat;
		a = a1; b = b1; c = c1; d = d1;
		norm = normalize((b - a) * (c - a));
	}
	Hit intersect(const Ray& ray) {
		Hit hit;
		Triangle topside(a, b, c, material);
		Triangle botside(b, c, d, material);
		Hit h1 = topside.intersect(ray);
		Hit h2 = botside.intersect(ray);
		if (h1.t > 0) {
			hit = h1;
		}
		else if (h2.t > 0) {
			hit = h2;
		}
		return hit;
	}
};

struct Cube :Intersectable{
	Triangle sides[12];

	Cube(vec3 r1, vec3 r2, vec3 r3, vec3 r4, vec3 r5, vec3 r6, vec3 r7, vec3 r8, Material* mat) {
		material = mat;
		////top
		//sides[0] = Side(r1, r2, r3, r4, mat);
		////back
		//sides[1] = Side(r1, r2, r5, r6, mat);
		////left
		//sides[2] = Side(r1, r4, r5, r8, mat);
		////right
		//sides[3] = Side(r2, r3, r6, r7, mat);
		////bottom
		//sides[4] = Side(r5, r6, r7, r8, mat);
		////front
		//sides[5] = Side(r4, r3, r8, r7, mat);

		//f  1//2  7//2  5//2
		sides[0] = Triangle(r1, r7, r5, mat);
		//f  1//2  3//2  7//2 
		sides[1] = Triangle(r1, r3, r7, mat);
		//f  1//6  4//6  3//6 
		sides[2] = Triangle(r1, r4, r3, mat);
		//f  1//6  2//6  4//6 
		sides[3] = Triangle(r1, r2, r4, mat);
		//f  3//3  8//3  7//3
		sides[4] = Triangle(r3, r8, r7, mat);
		//f  3//3  4//3  8//3 
		sides[5] = Triangle(r3, r4, r8, mat);
		//f  5//5  7//5  8//5 
		sides[6] = Triangle(r5, r7, r8, mat);
		//f  5//5  8//5  6//5 
		sides[7] = Triangle(r5, r8, r6, mat);
		//f  1//4  5//4  6//4 
		sides[8] = Triangle(r1, r5, r6, mat);
		//f  1//4  6//4  2//4 
		sides[9] = Triangle(r1, r6, r2, mat);
		//f  2//1  6//1  8//1 
		sides[10] = Triangle(r2, r6, r8, mat);
		//f  2//1  8//1  4//1 
		sides[11] = Triangle(r2, r8, r4, mat);

	}
	Cube(Material* mat) {
		Cube(vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f), vec3(0.0f, 1.0f, 0.0f),
			vec3(0.0f, 1.0f, 1.0f), vec3(1.0f, 0.0f, 0.0f), vec3(1.0f, 0.0f, 1.0f), vec3(1.0f, 1.0f, 0.0f), vec3(1.0f, 1.0f, 1.0f), mat);

	}

	Hit intersect(const Ray& ray) {
		Hit hit = sides[0].intersect(ray);
		for (int i = 0; i < 12; i++) {
			Hit h1 = sides[i].intersect(ray);
			if (h1.t > 0 && h1.t < hit.t) {
				hit = h1;
			}
		}
		return hit;
	}

};

class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

float rnd() { return (float)rand() / RAND_MAX; }

const float epsilon = 0.0001f;

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;
public:
	void build() {
		vec3 eye = vec3(1.5, 1, .5), vup = vec3(0, 1, 0), lookat = vec3(0, 0, .5);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.4f, 0.4f, 0.4f);
		vec3 lightDirection(1, 1, 1), Le(2, 2, 2);
		lights.push_back(new Light(lightDirection, Le));

		vec3 kd(0.3f, 0.2f, 0.1f), ks(2, 2, 2);
		Material* material = new Material(kd, ks, 50);
		Cube* c = new Cube(vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f), vec3(0.0f, 1.0f, 0.0f), 
			vec3(0.0f, 1.0f, 1.0f), vec3(1.0f, 0.0f, 0.0f), vec3(1.0f, 0.0f, 1.0f), vec3(1.0f, 1.0f, 0.0f), vec3(1.0f, 1.0f, 1.0f), material);
		objects.push_back(c);
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		//Original deleted, can be found in the 'minimal raytracing program'
		Hit firstHit = firstIntersect(ray);
		if (firstHit.t>0){
			float L = 0.2 * (1 + dot(firstHit.normal, ray.dir));
			printf("%f\n", L);
			return vec3(L, L, L);
		}
		else {
			return La;
		}
		
		
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}