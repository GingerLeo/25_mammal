data = readtable('\Parallel_Coordinates_Plot.xlsx'); 

unique_families = unique(data.Family);

fig = figure;

rows = 3;
cols = 3;

% Iterate over each family
for i = 1:numel(unique_families)
    family = unique_families{i};
    
    indices = strcmp(data.Family, family); 
    family_data = data(indices, :);
   
    species = family_data.Class_Label;
    family_data = family_data{:, 3:7};

    Labels={'Recognized','below 25', '25 to 50', '50 to 75','above 75'};
   
    subplot(rows, cols, i);
    
    % Create parallel coordinates plot
    parallelcoords(family_data, 'Group', species, 'labels', Labels, 'quantile', .25);
    
    ylabel('Pixels Percentage');
    
    ylim([0 1]);
    
    title(family);
end

filename = '\family_parallel_coordinates_combined.fig'; 
saveas(fig, filename);