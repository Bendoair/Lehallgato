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
	
	TriObject(std::vector<vec3> const vertices, std::vector<vec3> const sideurs) {
		for (vec3 siddefs : sideurs) {
			sides.push_back(Triangle(vertices[(int)siddefs.x], vertices[(int)siddefs.y], vertices[(int)siddefs.z]));
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
	//TODO
	//first multiply then move, add to vertecies, etc
	static TriObject* getObject(std::string const objFile, vec3 offset = vec3(0,0,0), float ratio) {
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

		return new TriObject(vertices, sideurs);

		
	}

};

//Icosaheder file
const char* IcosaObjFile = R"(v  0  -0.525731  0.850651
v  0.850651  0  0.525731
v  0.850651  0 - 0.525731
v - 0.850651  0 - 0.525731
v - 0.850651  0  0.525731
v - 0.525731  0.850651  0
v  0.525731  0.850651  0
v  0.525731 - 0.850651  0
v - 0.525731 - 0.850651  0
v  0 - 0.525731 - 0.850651
v  0  0.525731 - 0.850651
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


struct Icosahedron :Intersectable { //D20
	Triangle sides[20];
	Icosahedron(Material* mat, vec3 eltol, float ratio) {
		material = mat;
		vec3 v1 = vec3(0.0f, -0.525731, 0.850651);
		vec3 v2 = vec3(0.850651, 0.0f, 0.525731);
		vec3 v3 = vec3(0.850651, 0.0f, -0.525731);
		vec3 v4 = vec3(-0.850651, 0.0f, -0.525731);
		vec3 v5 = vec3(-0.850651, 0.0f, 0.525731);
		vec3 v6 = vec3(-0.525731, 0.850651, 0.0f);
		vec3 v7 = vec3(0.525731, 0.850651, 0.0f);
		vec3 v8 = vec3(0.525731, -0.850651, 0.0f);
		vec3 v9 = vec3(-0.525731, -0.850651, 0.0f);
		vec3 v10 = vec3(0.0f, -0.525731, -0.850651);
		vec3 v11 = vec3(0.0f, 0.525731, -0.850651);
		vec3 v12 = vec3(0.0f, 0.525731, 0.850651);

		v1 = (v1 + eltol) * ratio;
		v2 = (v2 + eltol) * ratio;
		v3 = (v3 + eltol) * ratio;
		v4 = (v4 + eltol) * ratio;
		v5 = (v5 + eltol) * ratio;
		v6 = (v6 + eltol) * ratio;
		v7 = (v7 + eltol) * ratio;
		v8 = (v8 + eltol) * ratio;
		v9 = (v9 + eltol) * ratio;
		v10 = (v10 + eltol) * ratio;
		v11 = (v11 + eltol) * ratio;
		v12 = (v12 + eltol) * ratio;

		sides[0] = Triangle(v2, v3, v7, mat);
		sides[1] = Triangle(v2, v8, v3, mat);
		sides[2] = Triangle(v4, v5, v6, mat);
		sides[3] = Triangle(v5, v4, v9, mat);
		sides[4] = Triangle(v7, v6, v12, mat);
		sides[5] = Triangle(v6, v7, v11, mat);
		sides[6] = Triangle(v10, v11, v3, mat);
		sides[7] = Triangle(v11, v10, v4, mat);
		sides[8] = Triangle(v8, v9, v10, mat);
		sides[9] = Triangle(v9, v8, v1, mat);
		sides[10] = Triangle(v12, v1, v5, mat);
		sides[11] = Triangle(v1, v12, v5, mat);
		sides[12] = Triangle(v7, v3, v11, mat);
		sides[13] = Triangle(v2, v7, v12, mat);
		sides[14] = Triangle(v4, v6, v11, mat);
		sides[15] = Triangle(v6, v5, v12, mat);
		sides[16] = Triangle(v3, v8, v10, mat);
		sides[17] = Triangle(v8, v2, v1, mat);
		sides[18] = Triangle(v4, v10, v9, mat);
		sides[19] = Triangle(v5, v9, v1, mat);

	}
	Hit intersect(const Ray& ray) {
		Hit hit = sides[0].intersect(ray);
		for (int i = 0; i < 20; i++) {
			Hit h1 = sides[i].intersect(ray);
			if (h1.t > 0 && (h1.t < hit.t || hit.t < 0)) {
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
	void pushDodecahedron() {
		vec3 corrig2 = vec3(7, 3, 0.8);
		float ratio2 = 8;
		//dodecahedron
		vec3 ddvert1 = (vec3(-.57735, -0.57735, 0.57735) + vec3(corrig2)) / ratio2;
		vec3 ddvert2 = (vec3(0.934172, 0.356822, 0) + vec3(corrig2)) / ratio2;
		vec3 ddvert3 = (vec3(0.93412, -0.356822, 0) + vec3(corrig2)) / ratio2;
		vec3 ddvert4 = (vec3(-0.934172, 0.356822, 0) + vec3(corrig2)) / ratio2;
		vec3 ddvert5 = (vec3(-0.93412, -0.356822, 0) + vec3(corrig2)) / ratio2;
		vec3 ddvert6 = (vec3(0, 0.934172, 0.356822) + vec3(corrig2)) / ratio2;
		vec3 ddvert7 = (vec3(0, 0.93412, -0.356822) + vec3(corrig2)) / ratio2;
		vec3 ddvert8 = (vec3(0.356822, 0, -0.934172) + vec3(corrig2)) / ratio2;
		vec3 ddvert9 = (vec3(-0.356822, 0, -0.934172) + vec3(corrig2)) / ratio2;
		vec3 ddvert10 = (vec3(0, -0.93412, -0.356822) + vec3(corrig2)) / ratio2;
		vec3 ddvert11 = (vec3(0, -0.934172, 0.356822) + vec3(corrig2)) / ratio2;
		vec3 ddvert12 = (vec3(0.356822, 0, 0.934172) + vec3(corrig2)) / ratio2;
		vec3 ddvert13 = (vec3(-0.356822, 0, 0.934172) + vec3(corrig2)) / ratio2;
		vec3 ddvert14 = (vec3(0.57735, 0.5775, -0.57735) + vec3(corrig2)) / ratio2;
		vec3 ddvert15 = (vec3(0.57735, 0.57735, 0.57735) + vec3(corrig2)) / ratio2;
		vec3 ddvert16 = (vec3(-0.57735, 0.5775, -0.57735) + vec3(corrig2)) / ratio2;
		vec3 ddvert17 = (vec3(-0.57735, 0.57735, 0.57735) + vec3(corrig2)) / ratio2;
		vec3 ddvert18 = (vec3(0.5775, -0.5775, -0.57735) + vec3(corrig2)) / ratio2;
		vec3 ddvert19 = (vec3(0.5775, -0.57735, 0.57735) + vec3(corrig2)) / ratio2;
		vec3 ddvert20 = (vec3(-0.5775, -0.5775, -0.57735) + vec3(corrig2)) / ratio2;


		Triangle* ddface1 = new Triangle(ddvert19, ddvert3, ddvert2);
		Triangle* ddface2 = new Triangle(ddvert12, ddvert19, ddvert2);
		Triangle* ddface3 = new Triangle(ddvert15, ddvert12, ddvert2);
		Triangle* ddface4 = new Triangle(ddvert8, ddvert14, ddvert2);
		Triangle* ddface5 = new Triangle(ddvert18, ddvert8, ddvert2);
		Triangle* ddface6 = new Triangle(ddvert3, ddvert18, ddvert2);
		Triangle* ddface7 = new Triangle(ddvert20, ddvert5, ddvert4);
		Triangle* ddface8 = new Triangle(ddvert9, ddvert20, ddvert4);
		Triangle* ddface9 = new Triangle(ddvert16, ddvert9, ddvert4);
		Triangle* ddface10 = new Triangle(ddvert13, ddvert17, ddvert4);
		Triangle* ddface11 = new Triangle(ddvert1, ddvert13, ddvert4);
		Triangle* ddface12 = new Triangle(ddvert5, ddvert1, ddvert4);
		Triangle* ddface13 = new Triangle(ddvert7, ddvert16, ddvert4);
		Triangle* ddface14 = new Triangle(ddvert6, ddvert7, ddvert4);
		Triangle* ddface15 = new Triangle(ddvert17, ddvert6, ddvert4);
		Triangle* ddface16 = new Triangle(ddvert6, ddvert15, ddvert2);
		Triangle* ddface17 = new Triangle(ddvert7, ddvert6, ddvert2);
		Triangle* ddface18 = new Triangle(ddvert14, ddvert7, ddvert2);
		Triangle* ddface19 = new Triangle(ddvert10, ddvert18, ddvert3);
		Triangle* ddface20 = new Triangle(ddvert11, ddvert10, ddvert3);
		Triangle* ddface21 = new Triangle(ddvert19, ddvert11, ddvert3);
		Triangle* ddface22 = new Triangle(ddvert11, ddvert1, ddvert5);
		Triangle* ddface23 = new Triangle(ddvert10, ddvert11, ddvert5);
		Triangle* ddface24 = new Triangle(ddvert20, ddvert10, ddvert5);
		Triangle* ddface25 = new Triangle(ddvert20, ddvert9, ddvert8);
		Triangle* ddface26 = new Triangle(ddvert10, ddvert20, ddvert8);
		Triangle* ddface27 = new Triangle(ddvert18, ddvert10, ddvert8);
		Triangle* ddface28 = new Triangle(ddvert9, ddvert16, ddvert7);
		Triangle* ddface29 = new Triangle(ddvert8, ddvert9, ddvert7);
		Triangle* ddface30 = new Triangle(ddvert14, ddvert8, ddvert7);
		Triangle* ddface31 = new Triangle(ddvert12, ddvert15, ddvert6);
		Triangle* ddface32 = new Triangle(ddvert13, ddvert12, ddvert6);
		Triangle* ddface33 = new Triangle(ddvert17, ddvert13, ddvert6);
		Triangle* ddface34 = new Triangle(ddvert13, ddvert1, ddvert11);
		Triangle* ddface35 = new Triangle(ddvert12, ddvert13, ddvert11);
		Triangle* ddface36 = new Triangle(ddvert19, ddvert12, ddvert11);

		objects.push_back(ddface1);
		objects.push_back(ddface2);
		objects.push_back(ddface3);
		objects.push_back(ddface4);
		objects.push_back(ddface5);
		objects.push_back(ddface6);
		objects.push_back(ddface7);
		objects.push_back(ddface8);
		objects.push_back(ddface9);
		objects.push_back(ddface10);
		objects.push_back(ddface11);
		objects.push_back(ddface12);
		objects.push_back(ddface13);
		objects.push_back(ddface14);
		objects.push_back(ddface15);
		objects.push_back(ddface16);
		objects.push_back(ddface17);
		objects.push_back(ddface18);
		objects.push_back(ddface19);
		objects.push_back(ddface20);
		objects.push_back(ddface21);
		objects.push_back(ddface22);
		objects.push_back(ddface23);
		objects.push_back(ddface24);
		objects.push_back(ddface25);
		objects.push_back(ddface26);
		objects.push_back(ddface27);
		objects.push_back(ddface28);
		objects.push_back(ddface29);
		objects.push_back(ddface30);
		objects.push_back(ddface31);
		objects.push_back(ddface32);
		objects.push_back(ddface33);
		objects.push_back(ddface34);
		objects.push_back(ddface35);
		objects.push_back(ddface36);
	}
	void build() {
		vec3 eye = vec3(2, 1.5, .5), vup = vec3(0, 0, 1), lookat = vec3(0, 0, .5);
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
		//Icosahedron* Ica = new Icosahedron(material,vec3(2,3, 0.850651), 0.25);
		TriObject* Ica = TriObject::getObject(IcosaObjFile);
		objects.push_back(Ica);
		pushDodecahedron();
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
		if (firstHit.t > 0) {
			float L = 0.2 * (1 + dot(normalize(firstHit.normal), -1 * normalize(ray.dir)));
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