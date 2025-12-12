// sca2012_lra.cpp - Long Range Attachments (SCA 2012)
// Implementation based on "Long Range Attachments - A Method to Simulate Inextensible Clothing in Computer Games"
// Using structure from previous hpbd.cpp

#if defined(WIN32)
#pragma warning(disable:4996)
#include <GL/freeglut.h>
#elif defined(__APPLE__) || defined(MACOSX)
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#define GL_SILENCE_DEPRECATION
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdio>

using glm::vec3;
using glm::length;
using glm::dot;
using glm::normalize;

// ---------------------------------------------------------
// Data Structures
// ---------------------------------------------------------

struct Particle {
    vec3 p;         // position
    vec3 old_p;     // previous position (for Verlet)
    vec3 v;         // velocity
    float w;        // inverse mass (0 = infinite mass/pinned)
    bool pinned = false;
};

// Standard PBD distance constraint (Local)
struct LocalConstraint {
    int i, j;
    float restLen;
};

// Long Range Attachment Constraint (Global)
struct LRAConstraint {
    int particleIdx;
    int attachmentIdx; // Index of the pinned particle used as anchor
    float maxDist;     // Initial geodesic (or euclidean in flat case) distance
};

// ---------------------------------------------------------
// Globals
// ---------------------------------------------------------

// Simulation Constants
static const float dt = 1.0f / 60.0f;
static const vec3 g(0.0f, -9.8f, 0.0f);
static const int clothW = 30;
static const int clothH = 30; // Taller to show stretching better
static const float spacing = 0.05f;

// Data
std::vector<Particle> P;
std::vector<LocalConstraint> localConstraints;
std::vector<LRAConstraint> lraConstraints;
std::vector<int> attachmentIndices; // Indices of pinned particles

// Parameters
int  g_iterations = 5;       // Low iteration count to demonstrate LRA benefit
bool g_useLRA = true;        // Toggle LRA
float g_lraSlack = 1.0f;     // 1.0 = exact length, 1.2 = 20% stretch allowed (Fig 5)

// Camera
float camDist = 3.5f;
float camYaw = 0.0f;
float camPitch = -15.0f * 3.14159265f / 180.0f;
glm::vec2 camPan(0.0f, -1.0f);
int lastMouseX = 0, lastMouseY = 0;
bool lbtn = false, rbtn = false;

// ---------------------------------------------------------
// Simulation Core
// ---------------------------------------------------------

inline int idx(int x, int y) { return y * clothW + x; }

void buildScene() {
    P.clear();
    P.resize(clothW * clothH);
    localConstraints.clear();
    lraConstraints.clear();
    attachmentIndices.clear();

    // 1. Init Particles
    for (int y = 0; y < clothH; ++y) {
        for (int x = 0; x < clothW; ++x) {
            int id = idx(x, y);
            // Center the cloth horizontally
            P[id].p = vec3((x - (clothW - 1) * 0.5f) * spacing,
                           (clothH - 1 - y) * spacing, 
                           0.0f);
            P[id].old_p = P[id].p;
            P[id].v = vec3(0.0f);
            
            // Pin top corners (Hanging Cloth setup)
            bool isPinned = (y == 0 && (x == 0 || x == clothW - 1));
            
            if (isPinned) {
                P[id].w = 0.0f;
                P[id].pinned = true;
                attachmentIndices.push_back(id);
            } else {
                P[id].w = 1.0f;
                P[id].pinned = false;
            }
        }
    }

    // 2. Build Local Constraints (Grid edges)
    auto addEdge = [&](int a, int b) {
        float d = length(P[a].p - P[b].p);
        localConstraints.push_back({a, b, d});
    };
    for (int y = 0; y < clothH; ++y) {
        for (int x = 0; x < clothW; ++x) {
            if (x + 1 < clothW) addEdge(idx(x, y), idx(x + 1, y));
            if (y + 1 < clothH) addEdge(idx(x, y), idx(x, y + 1));
        }
    }

    // 3. Build LRA Constraints
    // For every free particle, find the closest attachment point and store the initial distance.
    for (int i = 0; i < (int)P.size(); ++i) {
        if (P[i].pinned) continue;

        int bestAttach = -1;
        float minInitDist = 1e30f;

        // Simple strategy: Connect to the spatially closest attachment point in the rest configuration.
        // Since the mesh is initially flat, Euclidean distance == Geodesic distance.
        for (int attachID : attachmentIndices) {
            float d = length(P[i].p - P[attachID].p);
            if (d < minInitDist) {
                minInitDist = d;
                bestAttach = attachID;
            }
        }

        if (bestAttach != -1) {
            lraConstraints.push_back({i, bestAttach, minInitDist});
        }
    }
}

// Projection for Local Constraints (Standard PBD)
void projectLocal(const LocalConstraint& c) {
    Particle& p1 = P[c.i];
    Particle& p2 = P[c.j];
    
    vec3 dir = p1.p - p2.p;
    float dist = length(dir);
    if (dist < 1e-6f) return;
    
    float correction = (dist - c.restLen) * (1.0f - 0.0f /*stiffness=1*/); // simplified stiff
    vec3 grad = dir / dist;
    
    float wSum = p1.w + p2.w;
    if (wSum < 1e-6f) return;

    vec3 dp = -correction * grad;
    
    if (!p1.pinned) p1.p += dp * (p1.w / wSum);
    if (!p2.pinned) p2.p -= dp * (p2.w / wSum);
}

// Projection for LRA (The Core Algorithm)
void projectLRA(const LRAConstraint& c) {
    Particle& p = P[c.particleIdx];
    const Particle& attach = P[c.attachmentIdx];

    vec3 dir = p.p - attach.p;
    float currentDist = length(dir);
    
    // Apply Slack (Controlled Stretchiness, Section 3.5)
    float limit = c.maxDist * g_lraSlack;

    // Unilateral Constraint: Only project if stretched beyond limit
    if (currentDist > limit) {
        if (currentDist < 1e-6f) return; 
        
        // Project back to the surface of the sphere
        // p_new = center + dir * limit
        vec3 correction = dir * (limit / currentDist);
        p.p = attach.p + correction;
    }
}

void simulate() {
    // 1. Explicit Euler Integration (Prediction)
    for (auto& p : P) {
        if (p.pinned) continue;
        p.v += g * dt;
        p.old_p = p.p;
        p.p += p.v * dt;
    }

    // 2. Constraint Projection
    for (int iter = 0; iter < g_iterations; ++iter) {
        
        // (A) Local Constraints (Edges)
        // Maintain local shape / wrinkles
        for (const auto& c : localConstraints) {
            projectLocal(c);
        }

        // (B) LRA Constraints (Global Inextensibility)
        // Enforce global length limits immediately
        if (g_useLRA) {
            for (const auto& c : lraConstraints) {
                projectLRA(c);
            }
        }
    }

    // 3. Velocity Update & Damping
    for (auto& p : P) {
        if (p.pinned) continue;
        p.v = (p.p - p.old_p) / dt;
        p.v *= 0.99f; // Simple drag
    }
}

// ---------------------------------------------------------
// Visualization & UI
// ---------------------------------------------------------

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glTranslatef(camPan.x, camPan.y, -camDist);
    glRotatef(camPitch * 180.0f / 3.14159265f, 1, 0, 0);
    glRotatef(camYaw   * 180.0f / 3.14159265f, 0, 1, 0);

    // Draw Cloth Lines
    glColor3f(0.8f, 0.8f, 0.9f);
    glBegin(GL_LINES);
    for (const auto& c : localConstraints) {
        glVertex3fv(glm::value_ptr(P[c.i].p));
        glVertex3fv(glm::value_ptr(P[c.j].p));
    }
    glEnd();

    // Draw Points
    glPointSize(3.0f);
    glBegin(GL_POINTS);
    for(size_t i=0; i<P.size(); ++i) {
        if(P[i].pinned) glColor3f(1.0f, 0.2f, 0.2f); // Red for attachments
        else glColor3f(0.2f, 0.4f, 1.0f);            // Blue for free
        glVertex3fv(glm::value_ptr(P[i].p));
    }
    glEnd();
    
    // Optional: Draw LRA lines (faint green) to visualize attachments
    if (g_useLRA) {
        glColor3f(0.0f, 1.0f, 0.0f);
        glEnable(GL_BLEND); 
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glColor4f(0.2f, 1.0f, 0.2f, 0.15f); // Transparent
        glBegin(GL_LINES);
        for(const auto& c : lraConstraints) {
            // Only draw if significant tension? No, draw all to see topology
             glVertex3fv(glm::value_ptr(P[c.particleIdx].p));
             glVertex3fv(glm::value_ptr(P[c.attachmentIdx].p));
        }
        glEnd();
        glDisable(GL_BLEND);
    }

    glutSwapBuffers();
}

void idle() {
    simulate();
    
    // Performance title update
    static int frame = 0;
    static int t0 = 0;
    frame++;
    int t = glutGet(GLUT_ELAPSED_TIME);
    if (t - t0 > 200) {
        char buf[256];
        sprintf(buf, "SCA 2012 LRA Demo | LRA: %s | Slack: %.2f | Iters: %d", 
                g_useLRA ? "ON" : "OFF", g_lraSlack, g_iterations);
        glutSetWindowTitle(buf);
        t0 = t;
    }
    glutPostRedisplay();
}

void reshape(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, (double)w / h, 0.01, 100.0);
}

void mouseButton(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON)  lbtn = (state == GLUT_DOWN);
    if (button == GLUT_RIGHT_BUTTON) rbtn = (state == GLUT_DOWN);
    lastMouseX = x;
    lastMouseY = y;
}

void mouseMotion(int x, int y) {
    int dx = x - lastMouseX;
    int dy = y - lastMouseY;
    lastMouseX = x;
    lastMouseY = y;

    if (lbtn) {
        camYaw   += dx * 0.005f;
        camPitch += dy * 0.005f;
    }
    if (rbtn) {
        float s = 0.002f * camDist;
        camPan.x += dx * s;
        camPan.y -= dy * s;
    }
}

void keyboard(unsigned char key, int, int) {
    switch (key) {
    case 'l': case 'L':
        g_useLRA = !g_useLRA;
        printf("LRA: %s\n", g_useLRA ? "ON" : "OFF");
        break;
    case 'r': case 'R':
        buildScene();
        break;
    case ']': 
        g_lraSlack += 0.05f; 
        printf("Slack: %.2f\n", g_lraSlack);
        break;
    case '[': 
        g_lraSlack = std::max(1.0f, g_lraSlack - 0.05f); 
        printf("Slack: %.2f\n", g_lraSlack);
        break;
    case '1': g_iterations = 1; break;
    case '2': g_iterations = 2; break;
    case '3': g_iterations = 5; break;
    case '4': g_iterations = 10; break;
    case 27: exit(0); break;
    }
}

void usage() {
    printf("=== SCA 2012 Long Range Attachments Demo ===\n");
    printf("L       : Toggle LRA ON/OFF (Observe stretching without it!)\n");
    printf("R       : Reset Simulation\n");
    printf("[ / ]   : Decrease / Increase LRA Slack (Current: %.2f)\n", g_lraSlack);
    printf("1..4    : Set Iterations (Current: %d)\n", g_iterations);
    printf("Mouse   : Rotate (Left), Pan (Right)\n");
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(800, 600);
    glutCreateWindow("SCA 2012 LRA Cloth");

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);

    buildScene();
    usage();

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);
    glutMouseFunc(mouseButton);
    glutMotionFunc(mouseMotion);
    glutKeyboardFunc(keyboard);

    glutMainLoop();
    return 0;
}