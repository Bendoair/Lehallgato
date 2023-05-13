//=============================================================================================
// Computer Graphics Sample Program: Ray-tracing-let
//=============================================================================================
#include "framework.h"
#include <iostream>

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

struct Cone :Intersectable{
	vec3 point, norm, color;
	float height, alpha;

	
	Cone(vec3 _point, vec3 _normal, float _height, float _alpha, vec3 _color,
		Material* mat = new Material(vec3(0.3f, 0.2f, 0.1f), vec3(2, 2, 2), 50) ) {
		point = _point; norm = _normal; height = _height; alpha = _alpha; color = _color;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		float a = pow(dot(ray.dir, norm), 2) - dot(ray.dir, ray.dir) * pow(cosf(alpha), 2);
		float b = 2 * dot(ray.dir, norm)*dot((ray.start - point), norm)
			- (2*dot(ray.dir, (ray.start - point)) * pow(cosf(alpha), 2));
		float c = pow(dot((ray.start - point), norm), 2)
			- (dot((ray.start - point), (ray.start - point)) * pow(cosf(alpha), 2));


		float discriminant = b * b - 4 * a * c;
		float x1, x2;
		if (discriminant > 0) {
			x1 = (-b + sqrtf(discriminant)) / (2 * a);
			x2 = (-b - sqrtf(discriminant)) / (2 * a);
		}

		else if (discriminant == 0) {
			x1 = x2 = -b / (2 * a);
		}
		else {
			x1 = x2 = 0;
		}
		float t = -1;
		
		if (x1 > 0 && x2 > 0) {
			vec3 p1 = ray.start + ray.dir * x1;
			vec3 p2 = ray.start + ray.dir * x2;
			bool b1 = false;
			bool b2 = false;


			if (dot((p1 - point), norm) > 0) {
				b1 = true;
			}
			if (dot((p2 - point), norm) > 0) {
				b2 = true;
			}

			if (b1 && b2) {
				t = x1 < x2 ? x1 : x2;
				vec3 r = ray.start + ray.dir * t;
				if (dot((r - point), norm) > height) {
					t = x1 < x2 ? x2 : x1;
				}

			}
			else if (b1) {
				t = x1;
			}
			else if (b2) {
				t = x2;
			}
			vec3 r = ray.start + ray.dir * t;
			vec3 hitnormal = 2 * (dot((r-point),norm))*norm - ( 2 * (r - point)*pow(cosf(alpha), 2));

			hit.material = material;
			hit.normal = hitnormal; //correct? pls
			if (dot(hitnormal, ray.dir) > 0) {
				hit.normal = hit.normal * (-1);
			}
			hit.position = r;
			if (dot((hit.position - point), norm) >= 0 && dot((hit.position - point), norm) <= height) {
				hit.t = t;
			}
		}
		else if (x1 > 0 || x2 > 0) {
			t = x1>x2?x1:x2;
			
			vec3 r = ray.start + ray.dir * t;
			vec3 hitnormal = 2 * (dot((r - point), norm)) * norm - (2 * (r - point) * pow(cosf(alpha), 2));

			hit.material = material;
			hit.normal = hitnormal; //correct? pls
			if (dot(hitnormal, ray.dir) > 0) {
				hit.normal = hit.normal * (-1);
			}
			hit.position = r;
			if (dot((hit.position - point), norm) >= 0 && dot((hit.position - point), norm) <= height) {
				hit.t = t;
			}
		}
		
		
		return hit;
	}
};

struct Triangle :Intersectable {
	vec3 r1, r2, r3;
	vec3 norm;
	boolean reverse;

	Triangle(vec3 ia, vec3 ib, vec3 ic, Material* mat = new Material(vec3(0.3f, 0.2f, 0.1f), vec3(2, 2, 2), 50), boolean _reverse = false) {
		r1 = ia; r2 = ib; r3 = ic;
		norm = normalize(cross((r2 - r1), (r3 - r1)));
		reverse = _reverse;
		material = mat;
	}

	Triangle() {
		r1 = r2 = r3 = norm = vec3(0, 0, 0);
		material = new Material(vec3(0.3f, 0.2f, 0.1f), vec3(2, 2, 2), 50);
		reverse = false;
	}


	Hit intersect(const Ray& ray) {
		Hit hit;
		if (reverse)
		{
			if (dot(ray.dir, norm) < 0) {
				return hit;
			}
		}
		else {
			if (dot(ray.dir, norm) > 0) {
				return hit;
			}
		}
		float t = dot((r1 - ray.start), norm) / dot(ray.dir, norm);
		vec3 p = ray.start + ray.dir * t;
		if (dot(cross((r2 - r1), (p - r1)), norm) > 0
			&& dot(cross((r3 - r2), (p - r2)), norm) > 0
			&& dot(cross((r1 - r3), (p - r3)), norm) > 0) {
			//printf("lefutott a triangle if \n");
			hit.position = p;
			hit.normal = norm;
			hit.material = material;
			hit.t = t;
		}

		return hit;
	}
};

struct Cube :Intersectable {
	Triangle sides[12];

	Cube(vec3 r1, vec3 r2, vec3 r3, vec3 r4, vec3 r5, vec3 r6, vec3 r7, vec3 r8, Material* mat) {
		material = mat;

		//f  1//2  7//2  5//2 BOTTOM
		sides[0] = Triangle(r1, r7, r5, mat, true);
		//f  1//2  3//2  7//2 
		sides[1] = Triangle(r1, r3, r7, mat, true);
		//f  1//6  4//6  3//6 LEFT SIDE
		sides[2] = Triangle(r1, r4, r3, mat, true);
		//f  1//6  2//6  4//6 
		sides[3] = Triangle(r1, r2, r4, mat, true);
		//f  3//3  8//3  7//3 BACK SIDE
		sides[4] = Triangle(r3, r8, r7, mat, true);
		//f  3//3  4//3  8//3 
		sides[5] = Triangle(r3, r4, r8, mat, true);
		//f  5//5  7//5  8//5 
		sides[6] = Triangle(r5, r7, r8, mat, true);
		//f  5//5  8//5  6//5 
		sides[7] = Triangle(r5, r8, r6, mat, true);
		//f  1//4  5//4  6//4 
		sides[8] = Triangle(r1, r5, r6, mat, true);
		//f  1//4  6//4  2//4 
		sides[9] = Triangle(r1, r6, r2, mat, true);
		//f  2//1  6//1  8//1 
		sides[10] = Triangle(r2, r6, r8, mat, true);
		//f  2//1  8//1  4//1 
		sides[11] = Triangle(r2, r8, r4, mat, true);

	}
	Cube(Material* mat) {
		Cube(vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f), vec3(0.0f, 1.0f, 0.0f),
			vec3(0.0f, 1.0f, 1.0f), vec3(1.0f, 0.0f, 0.0f), vec3(1.0f, 0.0f, 1.0f), vec3(1.0f, 1.0f, 0.0f), vec3(1.0f, 1.0f, 1.0f), mat);

	}

	Hit intersect(const Ray& ray) {
		Hit hit = sides[0].intersect(ray);
		for (int i = 0; i < 12; i++) {
			Hit h1 = sides[i].intersect(ray);
			if (h1.t > 0 && (h1.t < hit.t || hit.t < 0)) {
				hit = h1;
			}
		}
		return hit;
	}

};

struct TriObject :Intersectable {
	std::vector<Triangle> sides;
	
	TriObject(std::vector<vec3> const vertices, std::vector<vec3> const sideurs, 
		float ratio, boolean reverse, vec3 offset = vec3(0.0, 0.0, 0.0)) {
		for (vec3 siddefs : sideurs) {
			sides.push_back(Triangle(
				vertices[(int)siddefs.x]*ratio + offset, 
				vertices[(int)siddefs.y] * ratio + offset,
				vertices[(int)siddefs.z] * ratio + offset)), true;
		}
	}

	Hit intersect(const Ray& ray ) {
		Hit hit;
		for (Triangle tri : sides) {
			Hit h1 = tri.intersect(ray);
			if (h1.t > 0 && (h1.t < hit.t || hit.t < 0)) {
				hit = h1;
			}
		}
		return hit;
		
	}
	static TriObject* getObject(std::string const objFile, float ratio, boolean reverse, vec3 offset = vec3(0.0, 0.0, 0.0)) {
		std::vector<std::string> lines;
		std::string currLine;
		for (int i = 0; i < objFile.size(); i++) {
			if (objFile[i] == '\n') {
				lines.push_back(currLine);
				currLine = "";
			}
			else {
				currLine = currLine + objFile[i];
			}
		}
		std::vector<vec3> vertices;
		std::vector<vec3> sideurs;
		for (std::string line : lines) {
			char id;
			vec3 data;
			sscanf(line.c_str(), "%c %f %f %f", &id, &data.x, &data.y, &data.z);
			if (id == 'v') {
				vertices.push_back(data);
			}
			else if (id == 'f') {
				sideurs.push_back(vec3(data.x-1, data.y-1, data.z-1));
			}
		}

		return new TriObject(vertices, sideurs, ratio, reverse, offset);

		
	}

};

//Icosaheder file
const char* IcosaObjFile = R"(v  0  -0.525731  0.850651
v  0.850651  0  0.525731
v  0.850651  0 -0.525731
v -0.850651  0 -0.525731
v -0.850651  0  0.525731
v -0.525731  0.850651  0
v  0.525731  0.850651  0
v  0.525731 -0.850651  0
v -0.525731 -0.850651  0
v  0 -0.525731 -0.850651
v  0  0.525731 -0.850651
v  0  0.525731  0.850651
f  2  3  7
f  2  8  3
f  4  5  6
f  5  4  9
f  7  6  12
f  6  7  11
f  10  11  3
f  11  10  4
f  8  9  10
f  9  8  1
f  12  1  2
f  1  12  5
f  7  3  11
f  2  7  12
f  4  6  11
f  6  5  12
f  3  8  10
f  8  2  1
f  4  10  9
f  5  9  1)";
//Dodecahedron file
const char* DodecaObjFile =
R"(v -0.57735 -0.57735  0.57735
v  0.934172  0.356822  0
v  0.934172 -0.356822  0
v -0.934172  0.356822  0
v -0.934172 -0.356822  0
v  0  0.934172  0.356822
v  0  0.934172 -0.356822
v  0.356822  0 -0.934172
v -0.356822  0 -0.934172
v  0 -0.934172 -0.356822
v  0 -0.934172  0.356822
v  0.356822  0  0.934172
v -0.356822  0  0.934172
v  0.57735  0.57735 -0.57735
v  0.57735  0.57735  0.57735
v -0.57735  0.57735 -0.57735
v -0.57735  0.57735  0.57735
v  0.57735 -0.57735 -0.57735
v  0.57735 -0.57735  0.57735
v -0.57735 -0.57735 -0.57735
f  19  3  2
f  12  19  2
f  15  12  2
f  8  14  2
f  18  8  2
f  3  18  2
f  20  5  4
f  9  20  4
f  16  9  4
f  13  17  4
f  1  13  4
f  5  1  4
f  7  16  4
f  6  7  4
f  17  6  4
f  6  15  2
f  7  6  2
f  14  7  2
f  10  18  3
f  11  10  3
f  19  11  3
f  11  1  5
f  10  11  5
f  20  10  5
f  20  9  8
f  10  20  8
f  18  10  8
f  9  16  7
f  8  9  7
f  14  8  7
f  12  15  6
f  13  12  6
f  17  13  6
f  13  1  11
f  12  13  11
f  19  12  11)";


class Camera {
	
public:
	vec3 eye, lookat, right, up;
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov = 45 * M_PI / 180) {
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
	
	void Animate(float dt) {
		eye = vec3((eye.x - lookat.x) * cos(dt) + (eye.z - lookat.z) * sin(dt) + lookat.x,
			eye.y,
			-(eye.x - lookat.x) * sin(dt) + (eye.z - lookat.z) * cos(dt) + lookat.z);
		set(eye, lookat, up);
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
public:
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	std::vector<Cone*> cones;
	Camera camera;
	vec3 La;

	void build() {
		vec3 eye = vec3(3, 2 , .5), vup = vec3(0, 0, 1), lookat = vec3(0, 0, .5);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.0f, 0.0f, 0.0f);
		vec3 lightDirection(1, 1, 1), Le(2, 2, 2);
		lights.push_back(new Light(lightDirection, Le));

		vec3 kd(0.3f, 0.2f, 0.1f), ks(2, 2, 2);
		Material* material = new Material(kd, ks, 50);
		vec3 v1 = vec3(0.0f, 0.0f, 0.0f);
		vec3 v2 = vec3(0.0f, 0.0f, 1.0f);
		vec3 v3 = vec3(0.0f, 1.0f, 0.0f);
		vec3 v4 = vec3(0.0f, 1.0f, 1.0f);
		vec3 v5 = vec3(1.0f, 0.0f, 0.0f);
		vec3 v6 = vec3(1.0f, 0.0f, 1.0f);
		vec3 v7 = vec3(1.0f, 1.0f, 0.0f);
		vec3 v8 = vec3(1.0f, 1.0f, 1.0f);
		Cube* cub = new Cube(v1, v2, v3, v4, v5, v6, v7, v8, material);
		objects.push_back(cub);
		TriObject* Ica = TriObject::getObject(IcosaObjFile, 0.15, false, vec3(0.8, 0.7, 0.15));
		objects.push_back(Ica);
		TriObject* Dca = TriObject::getObject(DodecaObjFile, 0.2, false, vec3(0.2, 0.7, 0.2));
		objects.push_back(Dca);

		Cone* cone1 = new Cone(vec3(0.5, 0.0, 0.2), vec3(0, 1, 0), 0.2, M_PI / 8, vec3(1,0,0));
		objects.push_back(cone1);
		cones.push_back(cone1);

	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
//#pragma omp parallel for
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
		if (firstHit.t > 0) {
			float L = 0.2 * (1 + dot(normalize(firstHit.normal), -1 * normalize(ray.dir)));
			vec3 colour = vec3(L, L, L);
			for (Cone* lehallgato : cones) {
				
				Ray seekray = Ray(firstHit.position, normalize((lehallgato->point + lehallgato->norm * epsilon) - firstHit.position));
				Hit refractionhit = firstIntersect(seekray);

				if (length(refractionhit.position - (lehallgato->point + lehallgato->norm * epsilon)) < epsilon) {
					colour = colour + lehallgato->color;
				}
				
				
			}
			return colour;
		}
		else {
			return La;
		}


	}
	void Animate(float dt) { camera.Animate(dt); }
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
//TODO
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') {
		scene.Animate(0.01f);
		glutPostRedisplay();
		onDisplay();
		//nem megy :((((((((
		
	}


	
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