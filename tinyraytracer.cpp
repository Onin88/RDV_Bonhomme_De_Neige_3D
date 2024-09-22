#define _USE_MATH_DEFINES
#include <cmath>
#include <limits>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <iomanip> // std::setprecision
#include <cstdlib> 

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "model.h"
#include "geometry.h"

// Taille de l'image de sortie
#define OUT_WIDTH 1280
#define OUT_HEIGHT 720

int envmap_width, envmap_height;
int snow_width, snow_height;
int wood_width, wood_height;
int scarf_width, scarf_height;
int noz_width, noz_height;

// Vecteurs de textures
std::vector<Vec3f> envmap;
std::vector<Vec3f> snow;
std::vector<Vec3f> wood;
std::vector<Vec3f> scarf;
std::vector<Vec3f> noz;

// Modèles 3D
Model hatModel("../assets/hat.obj");
Model leftArmModel("../assets/bras_gauche.obj");
Model rightArmModel("../assets/bras_droit.obj");
Model scarfModel("../assets/echarpe.obj");
Model buttonModel("../assets/bouton.obj");
Model nozModel("../assets/carotte.obj");

void afficherProgression(const std::string& titre, int valeurActuelle, int total) {
    constexpr int largeurBarre = 50;

    // Calculer le pourcentage de progression
    float pourcentage = static_cast<float>(valeurActuelle) / total;
    int barreRemplie = static_cast<int>(pourcentage * largeurBarre);

    // Effacer la ligne précédente
    std::cout << "\033[2K\r";

    // Afficher le titre et la barre de progression
    std::cout << titre << " : [";
    for (int i = 0; i < barreRemplie; ++i) {
        std::cout << ">";
    }
    for (int i = barreRemplie; i < largeurBarre; ++i) {
        std::cout << " ";
    }
    std::cout << "] " << std::fixed << std::setprecision(1) << pourcentage * 100 << "%" << std::flush;
}

void loadTexture(const std::string& filename, std::vector<Vec3f>& texture, int& width, int& height) {
    int channels = -1;
    unsigned char *texture_data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
    if (!texture_data || channels != 3) {
        std::cerr << "Erreur lors du chargement de la texture " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    texture = std::vector<Vec3f>(width*height);
    #pragma omp parallel for
    for (int j = height-1; j>=0 ; j--) {
        for (int i = 0; i<width; i++) {
            texture[i+j*width] = Vec3f(texture_data[(i+j*width)*3+0], texture_data[(i+j*width)*3+1], texture_data[(i+j*width)*3+2])*(1/255.);
        }
    }

    stbi_image_free(texture_data);
}

struct Light {
    Light(const Vec3f &p, const float i) : position(p), intensity(i) {}
    Vec3f position;
    float intensity;
};

struct Material {
    Material(const float r, const Vec4f &a, const Vec3f &color, const float spec) : refractive_index(r), albedo(a), diffuse_color(color), specular_exponent(spec) {}
    Material() : refractive_index(1), albedo(1,0,0,0), diffuse_color(), specular_exponent() {}
    float refractive_index;
    Vec4f albedo;
    Vec3f diffuse_color;
    float specular_exponent;
};

// Définition des materiaux
Material        ivory(1.0, Vec4f(0.6,  0.3, 0.1, 0.0), Vec3f(0.4, 0.4, 0.3),   50.);
Material        glass(1.5, Vec4f(0.0,  0.5, 0.1, 0.8), Vec3f(0.6, 0.7, 0.8),  125.);
Material   red_rubber(1.0, Vec4f(0.9,  0.1, 0.0, 0.0), Vec3f(0.3, 0.1, 0.1),   10.);
Material black_rubber(1.0, Vec4f(0.9,  0.1, 0.0, 0.0), Vec3f(0.1, 0.1, 0.1),   10.);
Material       mirror(1.0, Vec4f(0.0, 10.0, 0.8, 0.0), Vec3f(1.0, 1.0, 1.0), 1425.);
Material          white(1.0, Vec4f(1.0, 0.0, 0.0, 0.0), Vec3f(1.0, 1.0, 1.0), 50.0);

void applyTexture(const Vec3f &dir, std::vector<Vec3f>& texture, int& width, int& height, Material &material) {
    // ---- Calculs pour texturer l'objet ---- //
    // On calcul des 2 angles de direction du rayon
    float angle_x = atan2(dir.z, dir.x);
    float angle_y = asin(dir.y);

    // On calcul les coordonnées de l'envmap en fonction des angles en normalisant entre 0 et 1
    float coord_x = (angle_x + M_PI)/(2*M_PI);
    float coord_y = 1-(angle_y + M_PI/2)/M_PI; // On inverse l'image verticalement

    // On recupère les coordonnées sur l'image
    int x = std::min((int)(coord_x*width), width-1);
    int y = std::min((int)(coord_y*height), height-1);
    Vec3f textureTmp = texture[x+y*width];

    // On crée un materiel par rapport au pixel recuperé sur l'image de la texture
    material = Material(1.0, Vec4f(0.9,  0.1, 0.0, 0.0), Vec3f(textureTmp.x, textureTmp.y, textureTmp.z),   10.);
}

struct Sphere {
    Vec3f center;
    float radius;
    Material material;

    Sphere(const Vec3f &c, const float r, const Material &m) : center(c), radius(r), material(m) {}

    bool ray_intersect(const Vec3f &orig, const Vec3f &dir, float &t0) const {
        Vec3f L = center - orig;
        float tca = L*dir;
        float d2 = L*L - tca*tca;
        if (d2 > radius*radius) return false;
        float thc = sqrtf(radius*radius - d2);
        t0       = tca - thc;
        float t1 = tca + thc;
        if (t0 < 0) t0 = t1;
        if (t0 < 0) return false;
        return true;
    }
};

// Vecteurs d'objets
std::vector<Sphere> mouth;
std::vector<Sphere> eyes;
std::vector<Model> buttons;

Vec3f reflect(const Vec3f &I, const Vec3f &N) {
    return I - N*2.f*(I*N);
}

Vec3f refract(const Vec3f &I, const Vec3f &N, const float eta_t, const float eta_i=1.f) { // Snell's law
    float cosi = - std::max(-1.f, std::min(1.f, I*N));
    if (cosi<0) return refract(I, -N, eta_i, eta_t); // if the ray comes from the inside the object, swap the air and the media
    float eta = eta_i / eta_t;
    float k = 1 - eta*eta*(1 - cosi*cosi);
    return k<0 ? Vec3f(1,0,0) : I*eta + N*(eta*cosi - sqrtf(k)); // k<0 = total reflection, no ray to refract. I refract it anyways, this has no physical meaning
}

float distance_vector(Vec3f point1, Vec3f point2) {
    float deltaX = point2.x - point1.x;
    float deltaY = point2.y - point1.y;
    float deltaZ = point2.z - point1.z;
    return std::sqrt(deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ);
}

bool scene_intersect(const Vec3f &orig, const Vec3f &dir, const std::vector<Sphere> &spheres, Vec3f &hit, Vec3f &N, Material &material) {
    bool intersect = false;

    // Intersections des sphères du corps
    float spheres_dist = std::numeric_limits<float>::max();
    for (size_t i=0; i < spheres.size(); i++) {
        float dist_i;
        if (spheres[i].ray_intersect(orig, dir, dist_i) && dist_i < spheres_dist) {
            spheres_dist = dist_i;
            hit = orig + dir*dist_i;
            N = (hit - spheres[i].center).normalize();

            // On applique la texture de la neige
            applyTexture(dir, snow, snow_width, snow_height, material);

            intersect = true;
        }
    }

    intersect = false;

    // On affiche le chapeau => attention a la taille de .obj environ 2mm ici
    float hatModel_dist = std::numeric_limits<float>::max();
    for (int i=0; i<hatModel.nfaces(); i++) {
        if (intersect) {
            continue;
        }

        float dist_i;
        if (hatModel.ray_triangle_intersect(i, orig, dir, dist_i) && dist_i < hatModel_dist && dist_i < spheres_dist) {
            hatModel_dist = dist_i;
            hit = orig + dir*dist_i;
            N = cross(hatModel.point(hatModel.vert(i,1))-hatModel.point(hatModel.vert(i,0)), hatModel.point(hatModel.vert(i,2))-hatModel.point(hatModel.vert(i,0))).normalize();
            material = black_rubber;

            intersect = true;
        }
    }

    intersect = false;

    // On affiche le bras gauche => attention a la taille de .obj environ 4mm ici
    float leftArm_dist = std::numeric_limits<float>::max();
    for (int i=0; i<leftArmModel.nfaces(); i++) {
        if (intersect) {
            continue;
        }

        float dist_i;
        if (leftArmModel.ray_triangle_intersect(i, orig, dir, dist_i) && dist_i < leftArm_dist && dist_i < spheres_dist && dist_i < hatModel_dist) {
            leftArm_dist = dist_i;
            hit = orig + dir*dist_i;
            N = cross(leftArmModel.point(leftArmModel.vert(i,1))-leftArmModel.point(leftArmModel.vert(i,0)), leftArmModel.point(leftArmModel.vert(i,2))-leftArmModel.point(leftArmModel.vert(i,0))).normalize();
            
            // On applique la texture du bois
            applyTexture(dir, wood, wood_width, wood_height, material);

            intersect = true;
        }
    }

    intersect = false;

    // On affiche le bras droit => attention a la taille de .obj environ 4mm ici
    float rightArm_dist = std::numeric_limits<float>::max();
    for (int i=0; i<rightArmModel.nfaces(); i++) {
        if (intersect) {
            continue;
        }

        float dist_i;
        if (rightArmModel.ray_triangle_intersect(i, orig, dir, dist_i) && dist_i < rightArm_dist && dist_i < spheres_dist && dist_i < hatModel_dist && dist_i < leftArm_dist) {
            rightArm_dist = dist_i;
            hit = orig + dir*dist_i;
            N = cross(rightArmModel.point(rightArmModel.vert(i,1))-rightArmModel.point(rightArmModel.vert(i,0)), rightArmModel.point(rightArmModel.vert(i,2))-rightArmModel.point(rightArmModel.vert(i,0))).normalize();
            
            // On apllique la texture de bois
            applyTexture(dir, wood, wood_width, wood_height, material);

            intersect = true;
        }
    }

    intersect = false;

    // On affiche l'écharpe => attention a la taille de .obj environ 2.2mm ici
    float scarf_dist = std::numeric_limits<float>::max();
    for (int i=0; i<scarfModel.nfaces(); i++) {
        if (intersect) {
            continue;
        }

        float dist_i;
        if (scarfModel.ray_triangle_intersect(i, orig, dir, dist_i) && dist_i < scarf_dist && dist_i < spheres_dist && dist_i < hatModel_dist && dist_i < leftArm_dist && dist_i < rightArm_dist) {
            scarf_dist = dist_i;
            hit = orig + dir*dist_i;
            N = cross(scarfModel.point(scarfModel.vert(i,1))-scarfModel.point(scarfModel.vert(i,0)), scarfModel.point(scarfModel.vert(i,2))-scarfModel.point(scarfModel.vert(i,0))).normalize();
            
            // On applique la texture de la laine
            applyTexture(dir, scarf, scarf_width, scarf_height, material);

            intersect = true;
        }
    }

    intersect = false;

    // Intersections des boutons
    float buttons_dist = std::numeric_limits<float>::max();
    for (size_t i=0; i < buttons.size(); i++) {
        for (int j = 0; j < buttons[i].nfaces(); j++) {
            float dist_i;
            if (buttons[i].ray_triangle_intersect(j, orig, dir, dist_i) && dist_i < buttons_dist && dist_i < spheres_dist && dist_i < hatModel_dist && dist_i < leftArm_dist && dist_i < rightArm_dist && dist_i < scarf_dist) {
                buttons_dist = dist_i;
                hit = orig + dir*dist_i;
                N = cross(buttons[i].point(buttons[i].vert(j,1))-buttons[i].point(buttons[i].vert(j,0)), buttons[i].point(buttons[i].vert(j,2))-buttons[i].point(buttons[i].vert(j,0))).normalize();
                material = black_rubber;

                intersect = true;
            }
        }
    }

    // Intersection de la bouche
    float mouth_dist = std::numeric_limits<float>::max();
    for (size_t i=0; i < mouth.size(); i++) {
        float dist_i;
        if (mouth[i].ray_intersect(orig, dir, dist_i) && dist_i < mouth_dist && dist_i < spheres_dist && dist_i < hatModel_dist && dist_i < leftArm_dist && dist_i < rightArm_dist && dist_i < scarf_dist && dist_i < buttons_dist) {
            buttons_dist = dist_i;
            hit = orig + dir*dist_i;
            N = (hit - mouth[i].center).normalize(); 
            material = black_rubber;

            intersect = true;
        }
    }

    intersect = false;

    // On affiche l'écharpe => attention a la taille de .obj environ 4mm ici
    float noz_dist = std::numeric_limits<float>::max();
    for (int i=0; i<nozModel.nfaces(); i++) {
        if (intersect) {
            continue;
        }

        float dist_i;
        if (nozModel.ray_triangle_intersect(i, orig, dir, dist_i) && dist_i < noz_dist && dist_i < mouth_dist && dist_i < spheres_dist && dist_i < hatModel_dist && dist_i < leftArm_dist && dist_i < rightArm_dist && dist_i < scarf_dist && dist_i < buttons_dist) {
            noz_dist = dist_i;
            hit = orig + dir*dist_i;
            N = cross(nozModel.point(nozModel.vert(i,1))-nozModel.point(nozModel.vert(i,0)), nozModel.point(nozModel.vert(i,2))-nozModel.point(nozModel.vert(i,0))).normalize();
            
            // On applique la texture de la laine
            applyTexture(dir, noz, noz_width, noz_height, material);

            intersect = true;
        }
    }

    intersect = false;

    // Intersections des yeux
    float eyes_dist = std::numeric_limits<float>::max();
    for (size_t i=0; i < eyes.size(); i++) {
        float dist_i;
        if (eyes[i].ray_intersect(orig, dir, dist_i) && dist_i < eyes_dist && dist_i < noz_dist && dist_i < mouth_dist && dist_i < spheres_dist && dist_i < hatModel_dist && dist_i < leftArm_dist && dist_i < rightArm_dist && dist_i < scarf_dist && dist_i < buttons_dist) {
            eyes_dist = dist_i;
            hit = orig + dir*dist_i;
            N = (hit - eyes[i].center).normalize();
            material = black_rubber;

            intersect = true;
        }
    }

    // Intersection de sol plateau d'echec
    float checkerboard_dist = std::numeric_limits<float>::max();
    if (fabs(dir.y)>1e-3)  {
        float posY = -8.0; // Position Y du plan meme que le bonhomme de neige
        float d = (orig.y + posY)/dir.y; // the checkerboard plane has equation y = -4
        Vec3f pt = orig + dir*d;
        if (d>0 && fabs(pt.x)<20 && pt.z<0 && pt.z>-30 && d<spheres_dist && d<hatModel_dist && d<leftArm_dist && d<rightArm_dist && d<scarf_dist && d<buttons_dist && d<mouth_dist && d<noz_dist && d<eyes_dist) {
            checkerboard_dist = d;
            hit = pt;
            N = Vec3f(0,1,0);

            // On calcul des 2 angles de direction du rayon pour l'envmap
            float angle_x = atan2(dir.z, dir.x);
            float angle_y = asin(dir.y);

            // On calcul les coordonnées de l'envmap en fonction des angles en normalisant entre 0 et 1
            float coord_x = (angle_x + M_PI)/(2*M_PI);
            float coord_y = 1-(angle_y + M_PI/2)/M_PI; // On inverse l'image verticalement

            // On récupère le pixel de l'envmap correspondant aux coordonnées
            int x = std::min((int)(coord_x*envmap_width), envmap_width-1);
            int y = std::min((int)(coord_y*envmap_height), envmap_height-1);
            Vec3f textureTmp = envmap[x+y*envmap_width];

            // Calculer la valeur normalisée de pt.x dans la plage [0, 1]
            float normalizedX = (d - (-30)) / (0 - (-30));

            // Ajuster la plage de sortie de 0 à 0.1
            float coeff = 0.0;
            if (pt.x < 0.0) {
                coeff = normalizedX * 0.1;
            } else {
                coeff = normalizedX * 0.1 - pt.x/120;
            }

            // On crée un materiel par rapport au pixel recuperé sur l'image de la texture
            material.diffuse_color = Vec3f(textureTmp.x + coeff, textureTmp.y + coeff, textureTmp.z + coeff);
        }
    }

    return  std::min({spheres_dist, hatModel_dist, leftArm_dist, rightArm_dist, scarf_dist, buttons_dist, mouth_dist, noz_dist, eyes_dist, checkerboard_dist}) < 1000;
}

Vec3f cast_ray(const Vec3f &orig, const Vec3f &dir, const std::vector<Sphere> &spheres, const std::vector<Light> &lights, size_t depth=0) {
    Vec3f point, N;
    Material material;

    if (depth>4 || !scene_intersect(orig, dir, spheres, point, N, material)) {
        // On calcul des 2 angles de direction du rayon pour l'envmap
        float angle_x = atan2(dir.z, dir.x);
        float angle_y = asin(dir.y);

        // On calcul les coordonnées de l'envmap en fonction des angles en normalisant entre 0 et 1
        float coord_x = (angle_x + M_PI)/(2*M_PI);
        float coord_y = 1-(angle_y + M_PI/2)/M_PI; // On inverse l'image verticalement

        // On récupère le pixel de l'envmap correspondant aux coordonnées
        int x = std::min((int)(coord_x*envmap_width), envmap_width-1);
        int y = std::min((int)(coord_y*envmap_height), envmap_height-1);
        return envmap[x+y*envmap_width];
    }

    Vec3f reflect_dir = reflect(dir, N).normalize();
    Vec3f refract_dir = refract(dir, N, material.refractive_index).normalize();
    Vec3f reflect_orig = reflect_dir*N < 0 ? point - N*1e-3 : point + N*1e-3; // offset the original point to avoid occlusion by the object itself
    Vec3f refract_orig = refract_dir*N < 0 ? point - N*1e-3 : point + N*1e-3;
    Vec3f reflect_color = cast_ray(reflect_orig, reflect_dir, spheres, lights, depth + 1);
    Vec3f refract_color = cast_ray(refract_orig, refract_dir, spheres, lights, depth + 1);

    float diffuse_light_intensity = 0, specular_light_intensity = 0;
    for (size_t i=0; i<lights.size(); i++) {
        Vec3f light_dir      = (lights[i].position - point).normalize();
        float light_distance = (lights[i].position - point).norm();

        Vec3f shadow_orig = light_dir*N < 0 ? point - N*1e-3 : point + N*1e-3; // checking if the point lies in the shadow of the lights[i]
        Vec3f shadow_pt, shadow_N;
        Material tmpmaterial;
        if (scene_intersect(shadow_orig, light_dir, spheres, shadow_pt, shadow_N, tmpmaterial) && (shadow_pt-shadow_orig).norm() < light_distance)
            continue;

        diffuse_light_intensity  += lights[i].intensity * std::max(0.f, light_dir*N);
        specular_light_intensity += powf(std::max(0.f, -reflect(-light_dir, N)*dir), material.specular_exponent)*lights[i].intensity;
    }
    return material.diffuse_color * diffuse_light_intensity * material.albedo[0] + Vec3f(1., 1., 1.)*specular_light_intensity * material.albedo[1] + reflect_color*material.albedo[2] + refract_color*material.albedo[3];
}

void render(const std::vector<Sphere> &spheres, const std::vector<Light> &lights) {
    const float fov_deg  = 240.0; // FOV en degrés
    const float fov_rad  = fov_deg * M_PI / 180.0; // Conversion degrés -> radians
    //const float fov_rad = 4.8; // FOV en radians
    float dir_z = OUT_HEIGHT / (2. * tan(fov_rad / 2.));

    // On affiche la profondeur de champ
    std::cout << "Profondeur de champ : " << dir_z << std::endl;

    std::vector<unsigned char> pixmap(OUT_WIDTH * OUT_HEIGHT * 3);

    // Affichage du pourcentage de chargement
    float index = 0;
    #pragma omp parallel for
    for (size_t j = 0; j < OUT_HEIGHT; j++) { // actual rendering loop
        for (size_t i = 0; i < OUT_WIDTH; i++) {
            float dir_x = (i + 0.5) - OUT_WIDTH / 2.;
            float dir_y = -(j + 0.5) + OUT_HEIGHT / 2.; // this flips the image at the same time

            // Calcul de la couleur du pixel
            Vec3f color = cast_ray(Vec3f(0, 0, 0), Vec3f(dir_x, dir_y, dir_z).normalize(), spheres, lights);

            // Calcul du maximum des composantes du vecteur
            float max_component = std::max({color[0], color[1], color[2]});

            // Conversion et sauvegarde dans le pixmap
            size_t index_pixmap = (j * OUT_WIDTH + i) * 3;
            for (size_t k = 0; k < 3; k++) {
                float channel = max_component > 1 ? color[k] / max_component : color[k];
                pixmap[index_pixmap + k] = static_cast<unsigned char>(255 * std::max(0.f, std::min(1.f, channel)));
            }
        }

        // Affichage du pourcentage de chargement
        #pragma omp atomic
        index++;
        afficherProgression("Rendu de l'image", index, OUT_HEIGHT);
    }

    std::cout << std::endl; // Saut de ligne à la fin

    // Écriture de l'image dans un fichier
    stbi_write_jpg("out.jpg", OUT_WIDTH, OUT_HEIGHT, 3, pixmap.data(), 100);

    std::cout << "Terminé. Fichier enregistré : /build/out.jpg" << std::endl;
}


int main() {
    // Chargement des textures
    loadTexture("../assets/envmap.jpg", envmap, envmap_width, envmap_height); // Map de fond
    loadTexture("../assets/texture_snow.jpg", snow, snow_width, snow_height); // Texture de la neige
    loadTexture("../assets/texture_wood.jpg", wood, wood_width, wood_height); // Texture du bois
    loadTexture("../assets/texture_carotte.jpg", noz, noz_width, noz_height); // Texture de la carotte
    loadTexture("../assets/texture_laine.jpg", scarf, scarf_width, scarf_height); // Texture de la laine

    // Position du bonhomme de neige et taille
    float posX = 0.0, posY = -6.0, posZ = -6.0, size = 1;

    // Définition du corps
    std::vector<Sphere> spheres;
    spheres.push_back(Sphere(Vec3f(posX, posY, posZ), size + 1.0, white)); // base
    spheres.push_back(Sphere(Vec3f(posX, posY + 2.5, posZ), size + 0.5, white)); // corps
    spheres.push_back(Sphere(Vec3f(posX, posY + 4.5, posZ), size, white)); // tete

    // Définition de la bouche
    mouth.push_back(Sphere(Vec3f(posX - 0.5, posY + 4.3, posZ + size - 0.02),0.1 , black_rubber));
    mouth.push_back(Sphere(Vec3f(posX - 0.25, posY + 4.15, posZ + size - 0.05),0.1 , black_rubber));
    mouth.push_back(Sphere(Vec3f(posX, posY + 4.1, posZ + size - 0.05),0.1 , black_rubber));
    mouth.push_back(Sphere(Vec3f(posX + 0.25, posY + 4.15, posZ + size - 0.05),0.1 , black_rubber));
    mouth.push_back(Sphere(Vec3f(posX + 0.5, posY + 4.3, posZ + size - 0.02),0.1 , black_rubber));

    // Définition des yeux
    eyes.push_back(Sphere(Vec3f(posX - 0.3, posY + 4.8, posZ + size - 0.02), 0.1, black_rubber)); // oeil gauche
    eyes.push_back(Sphere(Vec3f(posX + 0.3, posY + 4.8, posZ + size - 0.02), 0.1, black_rubber)); // oeil droits

    // Définition des boutons
    Model btn1 = buttonModel;
    Model btn2 = buttonModel;
    Model btn3 = buttonModel;
    Model btn4 = buttonModel;

    // Position des boutons
    btn1.translate(posX - 0.25, posY + 3.4,posZ + size*2 - 0.8); // milieu
    btn2.translate(posX - 0.25, posY + 2.4, posZ + size*2 - 0.52); // milieu
    btn3.translate(posX - 0.25, posY + 1.25, posZ + size*2 - 0.52); // bas
    btn4.translate(posX - 0.25, posY + 0.25, posZ + size*2 - 0.07); // bas

    // Ajout des boutons dans le vecteur
    buttons.push_back(btn1);
    buttons.push_back(btn2);
    buttons.push_back(btn3);
    buttons.push_back(btn4);

    // position du chapeau
    hatModel.translate(posX - size, posY + 5.4, posZ - 1.0);

    // position du bras gauche
    leftArmModel.translate(posX - size * 4.5, posY, posZ);

    // position du bras droit
    rightArmModel.translate(posX, posY + 2.5, posZ);

    // position de l'écharpe
    scarfModel.translate(posX - 0.8, posY + 3.0, posZ - 0.5);

    // Position du nez
    nozModel.translate(posX - 0.1, posY + 4.4, posZ + size);

    // Position des lumières
    float lightX = 100.0, lightY = 50.0, lightZ = 100.0;

    // Définition des lumières, vec position (x, y, z), intensité
    std::vector<Light>  lights;
    lights.push_back(Light(Vec3f(lightX, lightY,  lightZ), 2.5)); // Soleil

    render(spheres, lights);

    return 0;
}