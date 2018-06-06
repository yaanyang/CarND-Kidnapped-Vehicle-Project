/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

// Create default_random_engine for later use
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    // Set number of particles
    num_particles = 100;

    // Initial weight
    double init_weight = 1.0;

    // Create normal (Gaussian) distributions for x, y and theta
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for (int i = 0; i < num_particles; ++i) {
        // New sample particle
		Particle sample_particle;
		
		// Sample from these normal distrubtions		
		sample_particle.id = i;
        sample_particle.x = dist_x(gen);
		sample_particle.y = dist_y(gen);
		sample_particle.theta = dist_theta(gen);
        sample_particle.weight = init_weight;

        particles.push_back(sample_particle);
        weights.push_back(init_weight);
    }

    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    for (unsigned int i = 0; i < particles.size(); ++i) {
        // If yaw_rate = 0
        if (fabs(yaw_rate) < numeric_limits<double>::epsilon()) {
            particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
        } 

        else {
            particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
            particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
            particles[i].theta += yaw_rate * delta_t;
        }

        // Create normal (Gaussian) distributions for x, y and theta
        normal_distribution<double> dist_x(0, std_pos[0]);
        normal_distribution<double> dist_y(0, std_pos[1]);
        normal_distribution<double> dist_theta(0, std_pos[2]);

        // Add random Gaussian noise
        particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    for (unsigned int i = 0; i < observations.size(); ++i) {
        // Set initial minimun distance to a large number
        double dist_min = numeric_limits<double>::max();

        for (unsigned int j = 0; j < predicted.size(); ++j) {
            // Current distance
            double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

            // If no assignment yet or smaller distance found, assign that landmark to this observation
            if (distance < dist_min) {
                dist_min = distance;
                observations[i].id = predicted[j].id;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    // Step through each particle
    for (unsigned int i = 0; i < particles.size(); ++i) {      
        
        // Hold observations in map's coordinates
        std::vector<LandmarkObs> observations_map;

        // Transform each observation from vehicle's coordinates to map's coordinates
        for (unsigned int j = 0; j < observations.size(); ++j) {  
            // Current observation          
            LandmarkObs obs_map;

            obs_map.x = particles[i].x + cos(particles[i].theta) * observations[j].x - sin(particles[i].theta) * observations[j].y;
            obs_map.y = particles[i].y + sin(particles[i].theta) * observations[j].x + cos(particles[i].theta) * observations[j].y;

            observations_map.push_back(obs_map);            
        }

        // Hold predicted measurement
        std::vector<LandmarkObs> predicted_meas;
        
        // Predicted measurement for each landmark with sensor range for this particle
        for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
            // Current prediction
            LandmarkObs pred_meas;

            double distance = dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);

            if (distance <= sensor_range) {
                pred_meas.id = map_landmarks.landmark_list[j].id_i;
                pred_meas.x = map_landmarks.landmark_list[j].x_f;
                pred_meas.y = map_landmarks.landmark_list[j].y_f;

                predicted_meas.push_back(pred_meas);
            }
        }

        // Data association
        dataAssociation(predicted_meas, observations_map);

        // Re-initialize particle weight
        particles[i].weight = 1.0;
                
        // Calculate new weight
        for (unsigned int j = 0; j < observations_map.size(); ++j) {
            //  Current observation
            double x_obs = observations_map[j].x;
            double y_obs = observations_map[j].y;
            double mu_x;
            double mu_y;

            // Find associated landmark
            for (unsigned int k = 0; k < predicted_meas.size(); ++k) {
                if (predicted_meas[k].id == observations_map[j].id) {
                    mu_x = predicted_meas[k].x;
                    mu_y = predicted_meas[k].y;
                }
            }

            // Calculate Multivariate-Gaussian Probability
            double std_x = std_landmark[0];
            double std_y = std_landmark[1];
            double gauss_norm = 1 / (2 * M_PI * std_x * std_y);
            double exponent= (pow((x_obs - mu_x), 2)) / (2 * pow(std_x, 2)) + (pow((y_obs - mu_y), 2)) / (2 * pow(std_y, 2));
            double obs_weight = gauss_norm * exp(-exponent);
            
            // Update final weight for each particle
            particles[i].weight *= obs_weight;            
        }
        // Store final weight for this particle
        weights[i] = particles[i].weight; 
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    // Hold new particles
    vector<Particle> new_particles (num_particles);

    // Generate discrete distribution propotional to particle weights
    discrete_distribution<int> dist_index (weights.begin(), weights.end());

    // Resampling
    for (int i = 0; i < num_particles; ++i) {
        new_particles[i] = particles[dist_index(gen)];
    }

    // Replace with new particles
    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
