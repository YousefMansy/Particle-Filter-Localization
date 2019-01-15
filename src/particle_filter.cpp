/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::uniform_int_distribution;
using std::uniform_real_distribution;

std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  // Set the number of particles
  num_particles = 20;

  // Create normal distributions for x, y and theta
  normal_distribution<double> dist_x(0, std[0]);
  normal_distribution<double> dist_y(0, std[1]);
  normal_distribution<double> dist_theta(0, std[2]);

  for (int i = 0; i < num_particles; i++)
  {
    Particle particle;

    // Initialize all particles to first position (based on estimates of x, y, theta and their uncertainties from GPS) and all weights to 1.
    particle.id = i;

    particle.x = x;
    particle.x += dist_x(gen); // add random gaussian noise

    particle.y = y;
    particle.y += dist_y(gen); // add random gaussian noise

    particle.theta = theta;
    particle.theta += dist_theta(gen);

    particle.weight = 1;

    particles.push_back(particle);
  }
  is_initialized = 1;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate)
{
  // Create normal distributions for x, y and theta
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; i++)
  {

    // Add measurements to each particle
    if (fabs(yaw_rate) < 0.00001)
    { // handle yaw_rate == 0
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    else
    { // handle yaw_rate != 0
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    // Add random gaussian noise
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs> &observations)
{
  for (int i = 0; i < observations.size(); i++)
  {
    LandmarkObs observation = observations[i];

    int id = -1;
    double min_dist = 9999999.99999;

    for (int j = 0; j < predicted.size(); j++)
    {
      LandmarkObs prediction = predicted[j];

      // calculate euclidian distance
      double distance = dist(observation.x, observation.y, prediction.x, prediction.y);

      // find nearest neighbour
      if (distance < min_dist && (min_dist = distance))
        id = prediction.id;
    }
    observations[i].id = id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks)
{
  for (int i = 0; i < num_particles; i++)
  {
    double x = particles[i].x;
    double y = particles[i].y;
    double theta = particles[i].theta;

    vector<LandmarkObs> landmarks;

    for (int j = 0; j < map_landmarks.landmark_list.size(); j++)
    {
      float landx = map_landmarks.landmark_list[j].x_f;
      float landy = map_landmarks.landmark_list[j].y_f;
      int landid = map_landmarks.landmark_list[j].id_i;

      // Check if distance from landmark is within sensor range
      if (dist(x, y, landx, landy) <= sensor_range)
        landmarks.push_back(LandmarkObs{landid, landx, landy});
    }

    // Transform observations from vehicle coordinates to map coordinates
    vector<LandmarkObs> t_observations;
    for (int j = 0; j < observations.size(); j++)
    {
      double tx = cos(theta) * observations[j].x - sin(theta) * observations[j].y + x;
      double ty = sin(theta) * observations[j].x + cos(theta) * observations[j].y + y;
      t_observations.push_back(LandmarkObs{observations[j].id, tx, ty});
    }

    // Associate between landmarks and observations
    dataAssociation(landmarks, t_observations);

    particles[i].weight = 1;

    for (int j = 0; j < t_observations.size(); j++)
    {
      double observationx = t_observations[j].x;
      double observationy = t_observations[j].y;

      double landx, landy;

      // Find association
      int association = t_observations[j].id;
      for (int m = 0; m < landmarks.size(); m++)
      {
        if (landmarks[m].id == association)
        {
          landx = landmarks[m].x;
          landy = landmarks[m].y;
          break;
        }
      }

      // Update weight
      double sigmax = std_landmark[0];
      double sigmay = std_landmark[1];
      double exponent = exp(-(pow(landx - observationx, 2) / (2 * pow(sigmax, 2)) + (pow(landy - observationy, 2) / (2 * pow(sigmay, 2)))));
      double weight = exponent / (2 * M_PI * sigmax * sigmay);
      particles[i].weight *= weight;
    }
  }
}

void ParticleFilter::resample()
{
  // Get weights
  vector<double> weights;
  for (int i = 0; i < num_particles; i++)
    weights.push_back(particles[i].weight);

  // Calculate maximum weight
  double max_weight = *max_element(weights.begin(), weights.end());

  // Initialize index randomly
  uniform_int_distribution<int> index_dist(0, num_particles - 1);
  int index = index_dist(gen);

  // Initialize beta and its distribution between 0 and double the maximum weight
  float beta = 0.0;
  uniform_real_distribution<double> beta_dist(0.0, max_weight*2);

  // Resampling wheel
  vector<Particle> p;
  for (int i = 0; i < num_particles; i++)
  {
    beta += beta_dist(gen);
    while (weights[index] < beta)
    {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    p.push_back(particles[index]);
  }

  // Update particles with resampled list
  particles = p;
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y)
{
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord)
{
  vector<double> v;

  if (coord == "X")
  {
    v = best.sense_x;
  }
  else
  {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}