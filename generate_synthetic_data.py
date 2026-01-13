import pandas as pd
import numpy as np
import random
from faker import Faker
import uuid

# Importar configuración centralizada de casillas AEAT
from tax_mapping_config import (
    TAX_MAPPING_CONFIG, 
    LEGACY_CASILLAS_MAP,
    get_prefixed_casilla
)

# Configuración
NUM_EMPRESAS = 10000  # ¡Cámbialo a 100.000 para pruebas de carga!
RATIO_FRAUDE = 0.05  # 5% de empresas fraudulentas
SEED = 42

fake = Faker('es_ES')
np.random.seed(SEED)
random.seed(SEED)

class TaxFraudSynthesizer:
    def __init__(self, num_companies, fraud_ratio):
        self.num_companies = num_companies
        self.fraud_ratio = fraud_ratio
        self.companies = []
        self.m200_data = []
        self.m347_data = []
        self.m349_data = []
        # Usar configuración centralizada AEAT
        self.casillas_map = LEGACY_CASILLAS_MAP

    def _generate_nif(self):
        letras = "ABCDEFGHJNPQRSUVW"
        return random.choice(letras) + str(random.randint(10000000, 99999999))

    def _generate_amount(self, base, variance=0.2, is_benford_compliant=True):
        """Genera importes realistas o manipulados (números redondos)"""
        if not is_benford_compliant:
            # Fraude: Números redondos o repetitivos
            return float(random.choice([10000, 15000, 20000, 50000, 99000]))
        
        # Real: Distribución Log-Normal (natural en finanzas)
        sigma = 0.5
        mu = np.log(base)
        val = np.random.lognormal(mu, sigma)
        return round(val, 2)

    def generate_ecosystem(self):
        print(f"Generando ecosistema de {self.num_companies} empresas...")
        
        # 1. Crear Empresas Base
        tipos = ['SANA'] * int(self.num_companies * (1 - self.fraud_ratio))
        tipos += ['FRAUDE_MAQUILLAJE'] * int(self.num_companies * (self.fraud_ratio / 2))
        tipos += ['FRAUDE_CARRUSEL'] * (self.num_companies - len(tipos))
        random.shuffle(tipos)

        for i in range(self.num_companies):
            empresa = {
                'id': i,
                'nif': self._generate_nif(),
                'nombre': fake.company(),
                'tipo': tipos[i],
                'sector_cnae': random.choice(['Construccion', 'Tecnologia', 'Comercio', 'Transporte'])
            }
            self.companies.append(empresa)

        # 2. Generar Datos Tributarios (Modelo 200 y 349)
        for emp in self.companies:
            self._fill_tax_forms(emp)

        # 3. Generar Red de Operaciones (Modelo 347)
        self._generate_transaction_network()

        return self._export_to_eav()

    def _fill_tax_forms(self, emp):
        # Base: Cifra de Negocio aleatoria entre 100k y 50M
        ventas_base = random.randint(100000, 50000000)
        
        if emp['tipo'] == 'SANA':
            # Coherencia contable
            caja = ventas_base * 0.10
            beneficio = ventas_base * 0.15
            gastos_fin = ventas_base * 0.02
            deuda = gastos_fin / 0.05  # 5% interés
            transporte = ventas_base * 0.05 if emp['sector_cnae'] in ['Comercio', 'Transporte'] else 0
            exportacion = ventas_base * 0.2 if random.random() > 0.7 else 0

        elif emp['tipo'] == 'FRAUDE_MAQUILLAJE':
            # Inflar beneficio, caja vacía, deuda oculta
            ventas_base *= 2 # Inflado
            caja = 1000 # Caja ridícula
            beneficio = ventas_base * 0.40 # Margen irreal
            deuda = ventas_base * 0.1 # Deuda declarada baja
            gastos_fin = deuda * 0.40 # Pero paga intereses como si tuviera mucha deuda (40% interés implícito)
            transporte = ventas_base * 0.02
            exportacion = 0

        elif emp['tipo'] == 'FRAUDE_CARRUSEL':
            # Mucha exportación, cero transporte
            exportacion = ventas_base * 0.9
            transporte = 0 # <--- ALERTA ROJA
            caja = 5000
            beneficio = 1000
            gastos_fin = 0
            deuda = 0

        # Rellenar Modelo 200 (Filas verticales)
        self.m200_data.append({'nif': emp['nif'], 'cod': self.casillas_map['cifra_negocio'], 'val': ventas_base})
        self.m200_data.append({'nif': emp['nif'], 'cod': self.casillas_map['resultado'], 'val': beneficio})
        self.m200_data.append({'nif': emp['nif'], 'cod': self.casillas_map['caja'], 'val': caja})
        self.m200_data.append({'nif': emp['nif'], 'cod': self.casillas_map['activo'], 'val': ventas_base * 1.5}) # Simplificado
        self.m200_data.append({'nif': emp['nif'], 'cod': self.casillas_map['deuda_bancaria'], 'val': deuda})
        self.m200_data.append({'nif': emp['nif'], 'cod': self.casillas_map['gastos_financieros'], 'val': gastos_fin})
        self.m200_data.append({'nif': emp['nif'], 'cod': self.casillas_map['gastos_transporte'], 'val': transporte})
        
        # Rellenar Modelo 349
        if exportacion > 0:
            self.m349_data.append({'nif': emp['nif'], 'cod': self.casillas_map['entregas_ue'], 'val': exportacion})

    def _generate_transaction_network(self):
        # Simula el grafo del Modelo 347
        nifs = [c['nif'] for c in self.companies]
        
        # A. Conexiones Aleatorias (Negocio Normal)
        num_transacciones = self.num_companies * 5
        for _ in range(num_transacciones):
            origen = random.choice(self.companies)
            destino = random.choice(self.companies)
            if origen['nif'] != destino['nif']:
                is_fraude = (origen['tipo'] != 'SANA')
                importe = self._generate_amount(50000, is_benford_compliant=not is_fraude)
                
                self.m347_data.append({
                    'nif_declarante': origen['nif'],
                    'nif_contraparte': destino['nif'],
                    'importe': importe,
                    'is_circular': False
                })

        # B. Inyectar Anillos de Fraude (Circularidad)
        # Tomamos las empresas de tipo CARRUSEL y hacemos anillos A->B->C->A
        carruseles = [c for c in self.companies if c['tipo'] == 'FRAUDE_CARRUSEL']
        
        # Crear grupos de 3 en 3
        for i in range(0, len(carruseles)-2, 3):
            emp_a = carruseles[i]
            emp_b = carruseles[i+1]
            emp_c = carruseles[i+2]
            
            monto_fijo = 500000.00 # Monto idéntico sospechoso
            
            # A -> B
            self.m347_data.append({'nif_declarante': emp_a['nif'], 'nif_contraparte': emp_b['nif'], 'importe': monto_fijo, 'is_circular': True})
            # B -> C
            self.m347_data.append({'nif_declarante': emp_b['nif'], 'nif_contraparte': emp_c['nif'], 'importe': monto_fijo, 'is_circular': True})
            # C -> A (Cierre del círculo)
            self.m347_data.append({'nif_declarante': emp_c['nif'], 'nif_contraparte': emp_a['nif'], 'importe': monto_fijo, 'is_circular': True})

    def _export_to_eav(self):
        # Convertimos todo al formato "Vertical" para Databricks
        vertical_rows = []
        
        # M200
        for row in self.m200_data:
            vertical_rows.append({'nif': row['nif'], 'modelo': '200', 'ejercicio': 2024, 'casilla': row['cod'], 'valor': row['val']})
            
        # M349
        for row in self.m349_data:
            vertical_rows.append({'nif': row['nif'], 'modelo': '349', 'ejercicio': 2024, 'casilla': row['cod'], 'valor': row['val']})
            
        df_vertical = pd.DataFrame(vertical_rows)
        df_m347 = pd.DataFrame(self.m347_data)
        df_empresas = pd.DataFrame(self.companies)
        
        return df_vertical, df_m347, df_empresas

# --- EJECUCIÓN ---
synthesizer = TaxFraudSynthesizer(num_companies=5000, fraud_ratio=0.10) # 5000 empresas, 10% fraude
df_eav, df_m347, df_info = synthesizer.generate_ecosystem()

print(f"\n--- DATASET GENERADO ---")
print(f"Total Registros Tributarios (Vertical): {len(df_eav)}")
print(f"Total Operaciones M347 (Grafo): {len(df_m347)}")
print(f"Total Empresas: {len(df_info)}")

# Vista previa de los datos
print("\nEjemplo Formato Vertical (Input Databricks):")
print(df_eav.head(5).to_string(index=False))

print("\nEjemplo M347 (Input Grafos):")
print(df_m347[df_m347['is_circular']==True].head(3).to_string(index=False))

# Guardar a CSV para uso posterior
df_eav.to_csv('data_eav.csv', index=False)
df_m347.to_csv('data_m347.csv', index=False)
df_info.to_csv('data_empresas.csv', index=False)
print("\nDatos guardados en: data_eav.csv, data_m347.csv, data_empresas.csv")
