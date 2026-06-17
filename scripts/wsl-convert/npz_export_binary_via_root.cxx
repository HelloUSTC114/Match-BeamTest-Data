// npz_export_binary_via_root.cxx
// WSL ROOT C++: read export.bin, write TTree with split-level branches
// Usage: root -l -q 'npz_export_binary_via_root.cxx("export.bin")'

#include "TFile.h"
#include "TTree.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>

struct DigiBlock {
    float amplitude[8];
    float polarity[8];
    float has_signal[8];
    float cfd50[8];
    float rise_time[8];
    float jitter[8];
    int   electrode[8];
    int   sensor_code[8];
};

struct OscBlock {
    float amplitude[6];
    float polarity[6];
    float has_signal[6];
    float cfd50[6];
    float rise_time[6];
    float jitter[6];
    int   electrode[6];
    int   sensor_code[6];
};

struct PositionBlock {
    float pos_M4, pos_BT4, pos_BT2;
    int   nch_M4, nch_BT4, nch_BT2;
};

void npz_export_binary_via_root(const char* bin_path) {
    FILE* fp = fopen(bin_path, "rb");
    if (!fp) { printf("ERROR: cannot open %s\n", bin_path); return; }

    int n_events = 0;
    fread(&n_events, sizeof(int), 1, fp);
    printf("Events: %d\n", n_events);

    // Determine output path: bin_path -> .root
    std::string out(bin_path);
    out.replace(out.rfind(".bin"), 4, ".root");
    TFile f(out.c_str(), "RECREATE");
    TTree tree("events", "AC-LGAD analysis");

    int   fpga_id;
    long  fpga_tick;
    DigiBlock digi;
    OscBlock  osc;
    PositionBlock pos;

    tree.Branch("fpga_id",   &fpga_id,   "fpga_id/I");
    tree.Branch("fpga_tick", &fpga_tick, "fpga_tick/L");
    tree.Branch("digi", &digi,
        "amplitude[8]/F:polarity[8]/F:has_signal[8]/F:cfd50[8]/F:rise_time[8]/F:jitter[8]/F:electrode[8]/I:sensor_code[8]/I");
    tree.Branch("osc", &osc,
        "amplitude[6]/F:polarity[6]/F:has_signal[6]/F:cfd50[6]/F:rise_time[6]/F:jitter[6]/F:electrode[6]/I:sensor_code[6]/I");
    tree.Branch("pos", &pos, "M4/F:BT4/F:BT2/F:nch_M4/I:nch_BT4/I:nch_BT2/I");

    for (int i = 0; i < n_events; i++) {
        // fpga metadata
        fread(&fpga_id,   sizeof(int),  1, fp);
        fread(&fpga_tick, sizeof(long), 1, fp);

        // digi: 8 ch x (5 float + 2 int)
        for (int j = 0; j < 8; j++) {
            fread(&digi.amplitude[j],   sizeof(float), 1, fp);
            fread(&digi.polarity[j],    sizeof(float), 1, fp);
            fread(&digi.has_signal[j],  sizeof(float), 1, fp);
            fread(&digi.cfd50[j],       sizeof(float), 1, fp);
            fread(&digi.rise_time[j],   sizeof(float), 1, fp);
            fread(&digi.jitter[j],      sizeof(float), 1, fp);
            fread(&digi.electrode[j],   sizeof(int),   1, fp);
            fread(&digi.sensor_code[j], sizeof(int),   1, fp);
        }
        // osc: 6 ch
        for (int j = 0; j < 6; j++) {
            fread(&osc.amplitude[j],   sizeof(float), 1, fp);
            fread(&osc.polarity[j],    sizeof(float), 1, fp);
            fread(&osc.has_signal[j],  sizeof(float), 1, fp);
            fread(&osc.cfd50[j],       sizeof(float), 1, fp);
            fread(&osc.rise_time[j],   sizeof(float), 1, fp);
            fread(&osc.jitter[j],      sizeof(float), 1, fp);
            fread(&osc.electrode[j],   sizeof(int),   1, fp);
            fread(&osc.sensor_code[j], sizeof(int),   1, fp);
        }
        // positions
        fread(&pos.pos_M4,  sizeof(float), 1, fp);
        fread(&pos.pos_BT4, sizeof(float), 1, fp);
        fread(&pos.pos_BT2, sizeof(float), 1, fp);
        fread(&pos.nch_M4,  sizeof(int),   1, fp);
        fread(&pos.nch_BT4, sizeof(int),   1, fp);
        fread(&pos.nch_BT2, sizeof(int),   1, fp);

        tree.Fill();
    }

    fclose(fp);
    tree.Write();
    f.Close();
    printf("Saved: %s\n", out.c_str());
}
