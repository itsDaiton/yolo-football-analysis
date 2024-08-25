from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
    
    def get_clustering_model(self, image):
        # Reshape the image to 2D
        image_2d = image.reshape(-1, 3)
        # Fit the KMeans model
        kmeans = KMeans(n_clusters=2, random_state=0, init="k-means++", n_init=1)
        kmeans.fit(image_2d)
        
        return kmeans
    
    def get_player_color(self, frame, bbox):
        # Get the image from the bounding box
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]    
        # Get the top half of the image  
        top_half_img = image[0:int(image.shape[0]/2), :]
        # Get the clustering model
        kmeans = self.get_clustering_model(top_half_img)    
        # Get the cluster labels
        labels = kmeans.labels_
        # Reshape the labels to the image shape
        clustered_img = labels.reshape(top_half_img.shape[0], top_half_img.shape[1])
        # Get the player cluster
        corner_clusters = [clustered_img[0, 0], clustered_img[0, -1], clustered_img[-1, 0], clustered_img[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster  
        # Get the player color based on the cluster
        player_color = kmeans.cluster_centers_[player_cluster]
        
        return player_color
    
    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            # Get the bounding box of the player
            bbox = player_detection['bbox']
            # Get the player color
            player_color = self.get_player_color(frame, bbox)
            # Append the player color to the list
            player_colors.append(player_color)
        
        # Fit the KMeans model to obtain the team colors
        kmeans = KMeans(n_clusters=2, random_state=0, init="k-means++", n_init=10)
        kmeans.fit(player_colors)
        
        self.kmeans = kmeans
        
        # Assign the team colors for each team
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]
        
    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1
        
        # Hardcode fix for player 108 (the model is not able to distinguish the player color correctly and it skews the results too much)
        if player_id == 103:
            team_id = 1
        
        self.player_team_dict[player_id] = team_id
        
        return team_id
        