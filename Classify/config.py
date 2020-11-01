import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms

transform = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                        ])

names23 = ['Container Ship',
            'Bulk Carrier',
            'Passengers Ship',
            'Ro-ro/passenger Ship',
            'Ro-ro Cargo',
            'Tug',
            'Vehicles Carrier',
            'Reefer',
            'Yacht',
            'Sailing Vessel',
            'Heavy Load Carrier',
            'Wood Chips Carrier',
            'Patrol Vessel',
            'Platform',
            'Standby Safety Vessel',
            'Combat Vessel',
            'Icebreaker',
            'Replenishment Vessel',
            'Tankers',
            'Fishing Vessels',
            'Supply Vessels',
            'Carrier/Floating',
            'Dredgers']

names26 = ['Container Ship', 'Bulk Carrier', 'Passengers Ship',
       'Ro-ro/passenger Ship', 'Ro-ro Cargo', 'Tug', 'Vehicles Carrier',
       'Reefer', 'Yacht', 'Sailing Vessel', 'Heavy Load Carrier',
       'Wood Chips Carrier', 'Livestock Carrier', 'Fire Fighting Vessel',
       'Patrol Vessel', 'Platform', 'Standby Safety Vessel',
       'Combat Vessel', 'Training Ship', 'Icebreaker',
       'Replenishment Vessel', 'Tankers', 'Fishing Vessels',
       'Supply Vessels', 'Carrier/Floating', 'Dredgers']