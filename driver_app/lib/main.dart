import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:google_fonts/google_fonts.dart';

void main() {
  runApp(const EVPSDriverApp());
}

class EVPSDriverApp extends StatelessWidget {
  const EVPSDriverApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'EVPS Driver',
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF1E1E1E),
        textTheme: GoogleFonts.robotoMonoTextTheme(ThemeData.dark().textTheme),
      ),
      home: const DriverDashboard(),
    );
  }
}

class DriverDashboard extends StatefulWidget {
  const DriverDashboard({super.key});

  @override
  State<DriverDashboard> createState() => _DriverDashboardState();
}

class _DriverDashboardState extends State<DriverDashboard> {
  // WebSocket Connection
  // Use 'localhost' for emulator/web, or your IP for real device
  final _channel = WebSocketChannel.connect(
    Uri.parse('ws://localhost:8000/ws'), 
  );

  // App State
  String currentEvId = "EV_0";
  double speed = 0.0;
  double eta = 0.0;
  bool isGreenWaveActive = false;
  String currentTls = "";
  LatLng evPosition = const LatLng(0, 0); // Default 0,0 until data comes
  bool hasData = false;

  final MapController _mapController = MapController();

  @override
  void dispose() {
    _channel.sink.close();
    super.dispose();
  }

  void _switchVehicle(String newId) {
    setState(() {
      currentEvId = newId;
      // Send command to Backend
      _channel.sink.add(jsonEncode({
        "type": "switch_ev",
        "ev_id": newId
      }));
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // --- APP BAR ---
      appBar: AppBar(
        title: const Text("EVPS PRIORITY SYSTEM"),
        backgroundColor: isGreenWaveActive ? Colors.green[800] : Colors.grey[900],
        elevation: 0,
        actions: [
          // Vehicle Switcher Dropdown
          DropdownButton<String>(
            value: currentEvId,
            dropdownColor: Colors.grey[800],
            underline: Container(),
            items: List.generate(50, (index) => "EV_$index")
                .map((id) => DropdownMenuItem(
                      value: id,
                      child: Text(id, style: const TextStyle(color: Colors.white)),
                    ))
                .toList(),
            onChanged: (val) {
              if (val != null) _switchVehicle(val);
            },
          ),
          const SizedBox(width: 20),
        ],
      ),

      // --- BODY ---
      body: StreamBuilder(
        stream: _channel.stream,
        builder: (context, snapshot) {
          if (snapshot.hasData) {
            _processData(snapshot.data);
          }

          return Stack(
            children: [
              // 1. MAP LAYER
              FlutterMap(
                mapController: _mapController,
                options: MapOptions(
                  // Must match REF_LAT/REF_LON from backend
                  initialCenter: const LatLng(51.5033, -0.1195), 
                  initialZoom: 16.0, // Zoom in closer
                ),
                children: [
                  TileLayer(
                    urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                    userAgentPackageName: 'com.evps.app',
                    // Dark Mode Filter for Map
                    tileBuilder: (context, widget, tile) {
                      return ColorFiltered(
                        colorFilter: const ColorFilter.mode(
                          Colors.black54, 
                          BlendMode.darken
                        ),
                        child: widget,
                      );
                    },
                  ),
                  if (hasData)
                    MarkerLayer(
                      markers: [
                        Marker(
                          point: evPosition,
                          width: 80,
                          height: 80,
                          child: const Icon(
                            Icons.local_hospital, // Ambulance Icon
                            color: Colors.blueAccent,
                            size: 40,
                          ),
                        ),
                      ],
                    ),
                ],
              ),

              // 2. DASHBOARD OVERLAY
              Positioned(
                bottom: 30,
                left: 20,
                right: 20,
                child: Container(
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: Colors.black87,
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(
                      color: isGreenWaveActive ? Colors.greenAccent : Colors.grey,
                      width: 2,
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: isGreenWaveActive 
                            ? Colors.green.withOpacity(0.5) 
                            : Colors.black.withOpacity(0.5),
                        blurRadius: 20,
                        spreadRadius: 5,
                      )
                    ],
                  ),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceAround,
                    children: [
                      // SPEED
                      _buildInfoColumn(
                        "SPEED", 
                        "${speed.toStringAsFixed(1)} km/h", 
                        Icons.speed
                      ),
                      // DIVIDER
                      Container(width: 1, height: 50, color: Colors.grey),
                      // ETA
                      _buildInfoColumn(
                        "ETA", 
                        "${eta.toStringAsFixed(1)} s", 
                        Icons.timer
                      ),
                    ],
                  ),
                ),
              ),

              // 3. GREEN WAVE ALERT (The "Wow" Factor)
              if (isGreenWaveActive)
                Positioned(
                  top: 100,
                  left: 20,
                  right: 20,
                  child: Container(
                    padding: const EdgeInsets.symmetric(vertical: 20),
                    decoration: BoxDecoration(
                      color: Colors.greenAccent.withOpacity(0.9),
                      borderRadius: BorderRadius.circular(15),
                    ),
                    child: Column(
                      children: [
                        const Icon(Icons.verified, color: Colors.black, size: 50),
                        const SizedBox(height: 10),
                        const Text(
                          "GREEN WAVE ACTIVE",
                          style: TextStyle(
                            color: Colors.black,
                            fontSize: 24,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        Text(
                          "Intersection $currentTls Cleared",
                          style: const TextStyle(
                            color: Colors.black87,
                            fontSize: 16,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
            ],
          );
        },
      ),
    );
  }

  Widget _buildInfoColumn(String label, String value, IconData icon) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, color: Colors.grey),
        const SizedBox(height: 5),
        Text(
          value,
          style: const TextStyle(
            fontSize: 28, 
            fontWeight: FontWeight.bold, 
            color: Colors.white
          ),
        ),
        Text(
          label,
          style: const TextStyle(color: Colors.grey, fontSize: 12),
        ),
      ],
    );
  }

  void _processData(dynamic data) {
    try {
      final decoded = jsonDecode(data);
      if (decoded['type'] == 'status') {
        setState(() {
          speed = decoded['speed'];
          eta = decoded['eta'];
          isGreenWaveActive = decoded['green_wave_active'];
          if (isGreenWaveActive) {
            currentTls = decoded['tls_id'];
          }
          
          // Map Updates
          double lat = decoded['lat'];
          double lon = decoded['lon'];
          // Basic SUMO->RealWorld Projection Fix
          // If SUMO returns 0,0 (Geo conversion failed), we use a fallback
          if (lat != 0 && lon != 0) {
             evPosition = LatLng(lat, lon);
             hasData = true;
             _mapController.move(evPosition, 16.0); // Follow vehicle
          }
        });
      }
    } catch (e) {
      print("Parse Error: $e");
    }
  }
}