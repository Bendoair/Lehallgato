//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Simon Benedek
// Neptun : CWTB5S
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	//uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec3 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x/(vp.z+1), vp.y/(vp.z+1), 0, 1);		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vao;	   // virtual world on the GPU

//Constants to vary for aesthetic pleasure
const int nvert = 1000; //number of verticves for a circle
const float baseSpeed = 0.006f; //Hami movement speed
const float baseRot = M_PI / 250.0f;
const int mainCharRotMult = 2;
const int mainCharSpeedMult = 1;
//KeyboardBool
bool pressed[256] = { false, }; // keyboard state



float dotH(vec3 a, vec3 b) {return a.x*b.x + a.y*b.y-a.z*b.z;}
float lengthH(vec3 a) { return sqrtf(dotH(a, a)); }
vec3 normalizeH(vec3 a) {return (a* (1 / lengthH(a)));}

//point that is actually in hyperbolic space
vec3 rePoint(vec3 point) {
	//return vec3(point.x, point.y, sqrtf(point.x * point.x + point.y * point.y + 1));
	float lambda = sqrtf(-1 / dotH(point, point));
	return lambda * point;


}



//vector the is actually in the hyperbolic space on a point in the hyperbolic space
vec3 reVec(vec3 vector, vec3 point /*on the hyperbolic space*/) {
	float lambda = dotH(point, vector);
	return vector + lambda * point;
}

//new point moving from a startpoint in a certain speed direction r(t)
vec3 newPoz(vec3 start, vec3 speed, float timeElapsed) {
	vec3 speedn = normalizeH(speed);
	return  rePoint(((start * coshf(timeElapsed) ) + (speedn * sinhf(timeElapsed))));
}
//new speed vector from a starting point after a certain amount of time elapsed
vec3 newSpeed(vec3 startPoint, vec3 startSpeed, float timeElapsed) {
	vec3 speedn = normalizeH(startSpeed);
	return reVec(startPoint * sinhf(timeElapsed) + speedn*coshf(timeElapsed), startPoint);
}

//hyperbolic distance of p and q
float distH(vec3 p, vec3 q) {
	return acoshf(-dotH(p, q));
}

//vector pointing from p to q
vec3 dir(vec3 p, vec3 q) {
	float t = distH(p, q);
	//return reVec((q - (p * coshf(t))) / sinhf(t), p);

	return ( - t * p * coshf(t) / sinhf(t)) + (q * t * coshf(0) / sinhf(t));
}

//point a certain distance away from an other point in a cetrain direction
vec3 pointThere(vec3 startPoint, vec3 direction, float dist) {
	return startPoint * coshf(dist) + normalizeH(direction) * sinhf(dist);
}

//egy adott pontban egy adott vektorra meroleges
vec3 meroleges(vec3 pont, vec3 mire) {
	return cross(vec3(pont.x, pont.y, -1 * pont.z), vec3(mire.x, mire.y, -1 * mire.z));
}

//rotate vector by certain degrees on a fixed point, returns new vector
vec3 rotby(vec3 point, vec3 startVector, float alpha) {
	return startVector * cosf(alpha) + meroleges(point, normalizeH(startVector)) * sinf(alpha);
}









class Circle {
private:
	vec3 center;
	float radius;
public:
	Circle(vec3 p, float r) { center = p; radius = r; }
	void drawCircle(vec3 startdir = vec3(1.0f, 0.0f, 0.3f)) {
		vec3 dir = normalizeH(reVec(startdir, center));
		vec3 vertices[nvert];
		for (int i = 0; i < nvert; i++) {
			dir = rotby(center, dir, 2 * M_PI / nvert);
			vertices[i] = pointThere(center, dir, radius);
		}
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec3) * nvert,  // # bytes
			vertices,	      	// address
			GL_DYNAMIC_DRAW);	// we do not change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			3, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed

		glBindVertexArray(vao);  // Draw call
		glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/, nvert /*# Elements*/);

		

	}
	void recenter(vec3 p) { center = p; }
	void resize(float r) { radius = r; }
};

class Goo {
private:
	std::vector<vec3> trackPoints;
public:
	Goo() { trackPoints = std::vector<vec3>(); }
	Goo(vec3 init) :Goo() { trackPoints.push_back(init); }
	void addPoint(vec3 point) { trackPoints.push_back(point); }
	void slime() {
		// Set color
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, 1.0f, 1.0f, 1.0f); // 3 floats
		
		
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			trackPoints.size() *sizeof(vec3),  // # bytes
			&trackPoints[0],	      	// address
			GL_DYNAMIC_DRAW);	// we do not change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			3, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed

		glBindVertexArray(vao);  // Draw call
		glDrawArrays(GL_LINE_STRIP, 0 /*startIdx*/, trackPoints.size()  /*# Elements*/);
	}
};
class Eyes {
private:
	vec3 lordCenter;
	vec3 lordDir;
	float lord_size;
	vec3 target;
public:
	Eyes(vec3 point,vec3 dir, float size, vec3 target) :lordCenter(point),lordDir(dir), lord_size(size), target(rePoint(target)) {}
	void drawEyes() {
		vec3 lefteyePoint = rePoint(pointThere(lordCenter, rotby(lordCenter, lordDir, -M_PI / 6), lord_size));
		vec3 righteyePoint = rePoint(pointThere(lordCenter, rotby(lordCenter, lordDir, M_PI / 6), lord_size));
		
		Circle left_white = Circle(lefteyePoint, lord_size / 3);
		Circle right_white = Circle(righteyePoint, lord_size / 3);
		
		vec3 leftTargetDir = dir(lefteyePoint, target);
		vec3 rightTargetDir = dir(righteyePoint, target);
		

		
		Circle left_iris = Circle(rePoint(pointThere(lefteyePoint, leftTargetDir, lord_size/8)), lord_size / 5);
		Circle right_iris = Circle(rePoint(pointThere(righteyePoint, rightTargetDir, lord_size / 8)), lord_size / 5);
		//Circle left_iris = Circle(lefteyePoint, lord_size / 6);
		//Circle right_iris = Circle(righteyePoint, lord_size / 6);
		
		
		//Color
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, 1.0f, 1.0f, 1.0f); // 3 floats
		left_white.drawCircle(target);
		right_white.drawCircle(target);

		//Color change
		glUniform3f(location, 0.0f, 0.0f, 1.0f); // 3 floats
		left_iris.drawCircle();
		right_iris.drawCircle();




	}
	void updateLord(vec3 center, vec3 dir) { lordCenter = center; lordDir = dir; }
	void targetAcquired(vec3 targettt) { target = targettt; }

};


class Hami {
private:
	Circle body;
	Circle mouth;
	Eyes eyes;
	Goo goop;
	vec3 direction;
	vec3 position;
	vec3 colour;
	float size;
public:
	Hami(vec3 startpoz, vec3 startdir, float initsize, vec3 colour):body(startpoz, initsize), colour(colour), 
		mouth(pointThere(startpoz, startdir, initsize), initsize/3), eyes(startpoz, startdir, initsize, vec3(0.0f,0.0f,1.0f)) {
		position = startpoz;
		direction = startdir;
		size = initsize;
		goop = Goo(startpoz);
	}
	void drawHami() {
		
		
		// Set color to (0, 1, 0) = green
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, colour.x, colour.y, colour.z); // 3 floats
		

		body.drawCircle(reVec(direction, position));

		// Set color to (0, 1, 0) = green
		glUniform3f(location, 0.0f,0.0f,0.0f); // 3 floats
		mouth.drawCircle();

		eyes.drawEyes();

	}
	void sloom() {
		goop.slime();
	}

	void move(float dt) {
		vec3 oldPoz = position;

		position = newPoz(position, direction, dt);
		body.recenter(position);
		mouth.recenter(pointThere(position, direction, size));
		direction = normalizeH(newSpeed(oldPoz, direction,  dt));
		goop.addPoint(oldPoz);
		eyes.updateLord(position, direction);
	}
	void rot(float phi) {
		direction = rotby(position, direction, phi);
		mouth.recenter(pointThere(position, direction, size));
		eyes.updateLord(position, direction);

	}
	void munch(float t) {
		mouth.resize((size/3) *fabs(sinf(t)));
	}
	void settarget(vec3 targ) {
		eyes.targetAcquired(targ);
	}
	
	vec3 getPoz() {
		return position;
	}

};


static void drawBackGround() {
	

	

	// Set color to (0, 1, 0) = green
	int location = glGetUniformLocation(gpuProgram.getId(), "color");
	glUniform3f(location, 0.0f, 0.0f, 0.0f); // 3 floats

	vec2 vertices[nvert];
	for (int i = 0; i < nvert; i++) {
		float phi = i * 2.0f * M_PI / nvert;
		vertices[i] = vec2(cosf(phi), sinf(phi));
	}
	glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
		sizeof(vec2) * nvert,  // # bytes
		vertices,	      	// address
		GL_DYNAMIC_DRAW);	// we do not change later

	glEnableVertexAttribArray(0);  // AttribArray 0
	glVertexAttribPointer(0,       // vbo -> AttribArray 0
		2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
		0, NULL); 		     // stride, offset: tightly packed

	glBindVertexArray(vao);  // Draw call
	glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/, nvert /*# Elements*/);



}





//The objects, whom will be in the hyperbolic space, ~vibing~
Hami redy = Hami(rePoint( vec3(0.0f, 0.0f, 1.0f)), reVec( vec3(1.0f, 0.0f, 0.0f), rePoint(vec3(0.5f, 0.5f, 1.0f)) ), 0.2f/*HamiSize*/, vec3(1.0f, 0.0f,0.0f));

Hami greeny = Hami(rePoint(vec3(0.89f, 0.03f, 1.0f)), reVec(vec3(-1.0f, 0.0f, 0.0f), rePoint(vec3(0.5f, 0.5f, 1.0f))), 0.2f/*HamiSize*/, vec3(0.0f, 1.0f, 0.0f));









// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	glGenVertexArrays(1, &vao);	// get 1 vao id
	glBindVertexArray(vao);		// make it active

	unsigned int vbo;		// vertex buffer object
	glGenBuffers(1, &vbo);	// Generate 1 buffer
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	// Geometry with 24 bytes (6 floats or 3 x 2 coordinates)

	redy.settarget(greeny.getPoz());
	greeny.settarget(redy.getPoz());

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.5f, 0.0f);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	drawBackGround();

	redy.settarget(greeny.getPoz());
	greeny.settarget(redy.getPoz());

	greeny.sloom();
	redy.sloom();

	greeny.drawHami();
	redy.drawHami();

	glutSwapBuffers(); // exchange buffers for double buffering
}


// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
	if (key == 'f') pressed['f'] = true;
	if (key == 's') pressed['s'] = true;
	if (key == 'e') pressed['e'] = true;
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
	if (key == 'f') pressed['f'] = false;
	if (key == 's') pressed['s'] = false;
	if (key == 'e') pressed['e'] = false;
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	//// Convert to normalized device space
	//float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	//float cY = 1.0f - 2.0f * pY / windowHeight;

	//char * buttonStat;
	//switch (state) {
	//case GLUT_DOWN: buttonStat = "pressed"; break;
	//case GLUT_UP:   buttonStat = "released"; break;
	//}

	//switch (button) {
	//case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	//case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	//case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	//}
}



// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	
	
	greeny.munch(time * 0.01f);
	greeny.move(baseSpeed);
	greeny.rot(baseRot);

	redy.munch(time * 0.01f);
	if (pressed['e'])redy.move(mainCharSpeedMult*baseSpeed);
	if (pressed['f'])redy.rot(mainCharRotMult*baseRot);
	if (pressed['s'])redy.rot(-baseRot* mainCharRotMult);

	
	
	
	
	glutPostRedisplay();
}
